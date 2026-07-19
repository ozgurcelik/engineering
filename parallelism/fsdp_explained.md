# FSDP From Scratch

## Introduction

The purpose of this project is to learn how FSDP works and implement it from scratch.

## Why FSDP?

FSDP stands for Fully Sharded Data Parallel.
Data parallel part refers to the data parallelism in the algorithm.
So, in each rank, we process a different subset of the data, and in that sense, it is works like the data parallel.
But one problem with the data parallellism is that each rank holds a full copy of the model, does a full forward and backward pass, and then syncs the gradients before the optimizer step.
This means that if a GPU cannot fit the model in its memory, then the data parallel approach is not feasible.
FSDP solves this problem by sharding the model across the ranks so that no rank needs to hold a full copy of the model.
Since FSDP shards the model parameters, we also shard the gradients and optimizer states across the ranks.
In a simplified mixed-precision analysis, a data-parallel baseline uses `(2 + 2 + K) * N_params` bytes of persistent model state per GPU: 2 bytes per BF16 model parameter, 2 bytes per BF16 gradient, and `K` bytes of optimizer-side state per parameter.
Here, `K` includes the FP32 master weight as well as the optimizer's other state.
For Adam with an FP32 master weight and FP32 first- and second-moment estimates, `K = 4 + 4 + 4 = 12`, giving 16 bytes per parameter in total.
If the Adam moments are stored in BF16 instead, `K = 4 + 2 + 2 = 8`, giving 12 bytes per parameter.
FSDP shards these persistent states, reducing this simplified estimate to `(2 + 2 + K) * N_params / N_gpus` bytes per GPU.
This is a general memory-motivation model rather than an exact accounting of our implementation: our persistent parameter shard is the FP32 master shard, and the lower-precision full parameter is materialized only for computation.
The estimate also omits activations, temporarily materialized full parameters and gradients, prefetching, padding, replicated parameters, and allocator overhead.

In our implementation, we will look at a simplified version of FSDP where we will assume a strictly linear and single-use execution order.
Additionally, we will be sharding each parameter separately.
By contrast, [PyTorch FSDP1](https://docs.pytorch.org/docs/stable/fsdp.html) concatenates the parameters managed by an FSDP unit into a `FlatParameter`.
[FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) uses per-parameter `DTensor` shards, but still groups parameters so that each group uses one all-gather and one reduce-scatter.


## How does FSDP work?

As we run the FSDP, each GPU will hold a shard of the model parameters.
But to be able to do the forward and backward pass through a layer in a GPU, we will need all the parameters for a layer materialized in the GPU.
This requires us to do an all-gather operation before the forward pass.
Once the forward pass is done, we can free the memory of the parameters.
Without prefetching, this means that during the forward pass we can get away with fully holding the parameters of only one layer at a time.
Prefetching deliberately materializes additional upcoming layers to overlap communication with computation, trading a higher transient memory peak for throughput.
During the backward pass, we will need all the parameters for the layer that we are currently processing once again.
Why is that?
Imagine we have a linear layer with parameters `W` and input `x` generating output `y`.
The forward pass is given by `y = Wx`.
And we have the loss function `L` with `dL/dy` available to us.
Then, the gradients for this layer is given by `dL/dW = dL/dy * x^T` and this does not need us to have the full `W` in memory.
But the input gradient `dL/dx = W^T * dL/dy` needs the full `W` when an input gradient is required.
Some backward paths do not need the parameter values: for example, a first linear layer whose input does not require a gradient, or an embedding backward, may not need `W` at all.
Our implementation nevertheless all-gathers every layer before backward because it uses a generic, layer-level mechanism rather than operator-specific knowledge.
And after doing the backward pass, we can free the memory of the parameters just like we did in the forward pass.
Once we have computed the gradients for a layer, we need to do reduce-scatter operation so that each GPU can get the gradients for its shard of the model.
For the full-sharding policy used here, where parameters are resharded after forward, each sharded parameter therefore participates in two all-gathers and one reduce-scatter per training iteration.
If the sharded parameters contain `N_params` elements and the world size is `P`, then, ignoring padding and the exact collective algorithm, the logical per-rank communication volume is approximately `3 * (P - 1) / P * N_params` elements.
A data-parallel all-reduce is equivalent to a reduce-scatter followed by an all-gather and communicates approximately `2 * (P - 1) / P * N_params` elements per rank.
Thus, under these assumptions, fully sharded data parallelism has 1.5 times the communication volume of data parallelism; `3 * N_params` versus `2 * N_params` is the large-`P` shorthand.
This comparison covers the sharded parameters only: our replicated parameters add their own gradient all-reduces, and our per-parameter collectives add more latency and padding overhead than grouped production implementations.

## The implementation

### Initializations

First of all, all ranks must agree on the initial weights and buffers, so we start by broadcasting from rank 0.

Then, we need to find all the layers that are going to be sharded.
Why are we not sharding everything?
Thats because some layers are not large enough to justify the communication overhead and latency costs.
In our implementation, we will be sharding the linear and embedding layers.
Every other trainable parameter will be replicated across all the GPUs.
These are typically normalization layer parameters.
In FSDP class initializer, we first define the lists `_fsdp_layers` and `_replicate_parameters` to store the layers and parameters that are going to be sharded and replicated respectively.
Then, we iterate over all the submodules of the model and check if it belongs to either of the lists.
```python
    def _find_fsdp_layers_and_replicated_parameters(self) -> None:
        """
        Find all the layers that are going to be sharded.
        Also find all the parameters that are going to be replicated.
        """
        for submodule in self.module.modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                self._fsdp_layers.append(submodule)
            else:
                for param in submodule.parameters(recurse=False):
                    if param.requires_grad:
                        self._replicate_parameters.append(param)
```
Doing this at module level helps us with keeping the logic simple.
For example, the weight and bias parameters of a linear layer are going to be visited together.
And since this loop visits every module in the model, the recurse=False avoids double counting.

Once we have collected the replicated parameters, we cast them to float32.
```python
    def _cast_replicated_params_to_float32(self) -> None:
        """
        Cast the replicated parameters to float32.
        """
        for param in self._replicate_parameters:
            param.data = param.data.to(torch.float32)
```
This mirrors the master-weight choice we make for the sharded parameters: even when we train in a lower compute dtype, the replicated parameters (typically normalization weights) are kept in float32 so that their gradients are accumulated and all-reduced in full precision.
This is important because normalization layers are numerically sensitive, and they are small enough that keeping a float32 copy costs us almost nothing.

Now, we need to do some bookkeeping.
For each layer, we will define a `FSDPLayerState` object to store everything we will need related to that layer.

```python
@dataclass
class ShardMetadata:
    num_elements: int
    shard_size: int
    padded_num_elements: int
    shape: torch.Size
    start: int
    end: int

@dataclass
class FSDPParamState:
    name: str
    metadata: ShardMetadata
    local_param: torch.nn.Parameter
    full_param: torch.nn.Parameter | None = None
    forward_gather_handle: dist.Work | None = None
    backward_gather_handle: dist.Work | None = None

@dataclass
class FSDPLayerState:
    layer: torch.nn.Module
    param_states: dict[str, FSDPParamState] = field(default_factory=dict)
    restore_dtype: torch.dtype | None = None
```
The `ShardMetadata` object lets us describe how the sharding of a parameter will be done.
To understand what is going on in it, it helps us to look at the code where we get the shard metadata.
```python
    def _get_shard_metadata(self, param: torch.nn.Parameter) -> ShardMetadata:
        """
        Get the metadata for the shard.
        This will be used to get the local shard, reconstruct the full parameter, and all-gather the parameter.
        """
        num_elements = param.numel()
        shard_size = (num_elements + self.world_size - 1) // self.world_size
        padded_num_elements = shard_size * self.world_size
        start = self.rank * shard_size
        end = start + shard_size
        return ShardMetadata(
            num_elements=num_elements,
            shard_size=shard_size,
            padded_num_elements=padded_num_elements,
            shape=param.shape,
            start=start,
            end=end,
        )
```
As we can see, the num_elements is the actual number of elements in that parameter.
But this number is not necessarily divisible by the number of GPUs, and we want all the GPUs to have the same number of elements in their shards.
So, we pad the number of elements to the next multiple of the number of GPUs.
This is how we get the shard size and padded number of elements, respectively.
Shape is the shape of the parameter and we keep it so that we can reconstruct the full parameter later.
`start` is the index of the first element in this rank's shard, while `end` is the exclusive endpoint of the half-open slice `[start, end)`.
These two will have different values for each rank.

In the ParamState object, we store the metadata for the parameter, the local parameter which is the shard of the parameter for this rank, and the full parameter which will be rematerialized later for the forward and backward passes and freed when we are done with it.
We also store the handles for the forward and backward gather operations which are done asynchronously.

In the LayerState object, we have the param_states dictionary to store the ParamState objects for each parameter in the layer.
Remember that linear layer has weight and bias parameters each with their own ParamState object.
Restore dtype records the activation-boundary dtype to which floating-point outputs are cast after the layer runs.
Normally this is the dtype of the first floating-point input.
For a layer such as an embedding whose inputs are integer indices, we use the dtype of its persistent FP32 master weight shard.

We populate the layer states like this:
```python
    def _get_local_shard_and_metadata(self, param: torch.nn.Parameter) -> tuple[torch.Tensor, ShardMetadata]:
        """
        Get the local shard of the parameter and the metadata.
        """
        metadata = self._get_shard_metadata(param)

        flattened_param = param.detach().flatten()

        if metadata.padded_num_elements > metadata.num_elements:
            padding = torch.zeros(metadata.padded_num_elements - metadata.num_elements, dtype=param.dtype, device=param.device)
            flattened_param = torch.cat([flattened_param, padding])

        # Master weight is always in float32.
        local_shard = flattened_param[metadata.start:metadata.end].clone().to(torch.float32)
        return local_shard, metadata

    def _create_layer_state(self, layer: torch.nn.Module) -> FSDPLayerState:
        """
        Create the state for the layer and set the local parameter to the layer.
        """
        param_states: dict[str, FSDPParamState] = {}
        for param_name in ("weight", "bias"):
            param = getattr(layer, param_name, None)
            if param is None:
                continue

            # inherit the requires_grad from the original parameter
            local_shard, metadata = self._get_local_shard_and_metadata(param)
            param_state = FSDPParamState(
                name=param_name,
                metadata=metadata,
                local_param=torch.nn.Parameter(local_shard, requires_grad=param.requires_grad),
            )
            param_states[param_name] = param_state
            setattr(layer, param_name, param_state.local_param)
            
        return FSDPLayerState(
            layer=layer,
            param_states=param_states,
        )

    def _create_layer_states(self) -> None:
        """
        Create the states for the FSDP layers.
        """
        for layer in self._fsdp_layers:
            self._layer_states[layer] = self._create_layer_state(layer)
```

We flatten and pad the parameter and then take the slice for this rank.
We always store the master weight in float32.
When `compute_dtype` is configured, the gathered full parameter and the full gradient produced by autograd use that lower precision, and the reduce-scatter also runs in that dtype.
After the reduce-scatter completes, we convert the reduced local gradient to the FP32 local-parameter dtype and attach it to, or accumulate it into, the FP32 master shard.
Thus the optimizer sees an FP32 local gradient, but the gradient computation and communication are not necessarily performed in FP32.
Also, we set the layer's parameters to the local parameter instead of the full parameter.

This local parameter is what closes the training loop.
The optimizer steps on the FP32 local shards (`local_param`), never on the full or compute-dtype parameters.
Each rank owns a disjoint shard, so each rank's optimizer only updates its own slice of the master weights, and there is no redundant work across ranks.
This is exactly why the optimizer states (for example Adam's moments) end up sharded too: they are created and kept per `local_param`, so each rank only stores the optimizer state for its own shard.
After the step, the updated FP32 values live in `local_param`, and they are re-cast to the compute dtype the next time we all-gather that layer, so the following forward pass automatically sees the freshly updated weights.

### Forward Pass

In the forward pass, we need to do an all-gather operation for the parameters of the layer before we can do the forward pass.
But doing this synchronously would add the full latency cost of the all-gather operation to the forward pass.
Instead, we will overlap the all-gather operation of a layer with the forward pass of the previous layer(s).
This is called prefetching.
So, when its the turn of a layer to do the forward pass, we will already have the parameters in memory.
This is enabled by multiple moving blocks of code.
We first define the all-gather operation as an asynchronous operation.
```python
    def _all_gather_param_async(self, local_param: torch.nn.Parameter, 
                            metadata: ShardMetadata) -> tuple[dist.Work | None, torch.Tensor]:
        """
        Issue an async all-gather operation.
        """
        local_shard = local_param.detach()
        if self.compute_dtype is not None:
            local_shard = local_shard.to(self.compute_dtype)
        if self.world_size == 1:
            return None, local_shard.clone()
        buffer = torch.empty(metadata.padded_num_elements, dtype=local_shard.dtype, device=local_shard.device)
        handle = dist.all_gather_into_tensor(buffer, local_shard, async_op=True)
        return handle, buffer
```
There we handle the case where the world size is 1 separately and remember to map the local shard to the compute dtype if we are using it.
We allocate a buffer to store the all-gathered parameter and issue the all-gather operation asynchronously.
So, the buffer's storage is fully allocated, but it just has uninitialized garbage data for the time being until the all-gather operation is completed and its populated with the actual data.
```python
    def _prefetch_layer_forward(self, layer: torch.nn.Module) -> None:
        """
        Issue an async all-gather operation for the layers params for the forward pass.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if self._storage_allocated(param_state.full_param):
                continue
            local_param = getattr(layer, param_name, None)
            metadata = param_state.metadata
            if local_param is None or metadata is None:
                continue
            handle, buffer = self._all_gather_param_async(local_param, metadata)
            # inherit the requires_grad from the original parameter and add the post-accumulate-grad hook if it requires grad
            # we add the hook here because hook needs to be attached to the full param before the forward pass.
            # if we were to wait till the prefetch backward pass, then the autograd would have already built the graph from forward pass
            # the hooks need to be attached to the object that participates in the forward pass.
            param_state.full_param = torch.nn.Parameter(buffer, requires_grad=local_param.requires_grad)
            if param_state.full_param.requires_grad:
                param_state.full_param.register_post_accumulate_grad_hook(self._make_reduce_scatter_hook(layer, param_name))
            param_state.forward_gather_handle = handle
```
In the prefetch function, we iterate over all the parameters of the layer (such as weight and bias of a linear layer) and check if the full parameter is already allocated.
If it is, we skip it.
Otherwise, we wrap the buffer in a Parameter object and attach the post-accumulate-grad hook if the parameter requires grad.
Had we done this after the forward pass, then the autograd graph would refer to the old object, and not to the tensor in the buffer.
This is also the reason why we add the post-accumulate-grad hook here which we will explain in the backward pass section.
We also store the handle for the all-gather operation so that we can reference to it later.
```python
    def _use_prefetched_layer_forward(self, layer: torch.nn.Module) -> None:
        """
        Wait for the all-gathers, trim the padding, reshape the parameter, and attach it to the layer.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if param_state.forward_gather_handle is not None:
                param_state.forward_gather_handle.wait()
            full_param = param_state.full_param
            full_param.data = full_param.data[:param_state.metadata.num_elements].view(param_state.metadata.shape)
            setattr(layer, param_name, full_param)
            param_state.forward_gather_handle = None
```
If the parameters have the forward gather handle, we wait for it to complete and then trim the padding and reshape the parameter to its original shape.
Note that its possible that the all-gather has already been completed by that time, so the wait() is basically instant, and thats the actual purpose of the prefetching.
But if the all-gather is not complete, then we wait for it to complete.
We then update the data of the full_param to the actual data and set the layer's parameter to the full_param.
Note that we are updating the data of the full_param alone, and not the parameter object itself.
So, the hook we attached to the full_param will still be there and will be called when the gradients are accumulated.
We also clear the forward gather handle so that we know that the all-gather is done.
```python
    def _pre_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        """
        Pre-forward hook for the FSDP layers.
        We should do the all-gather of the parameters for the forward pass.
        Also cast the inputs to the compute dtype.
        """
        layer_state = self._layer_states[layer]
        if self.compute_dtype is not None:
            master_dtype = layer_state.param_states["weight"].local_param.dtype
            layer_state.restore_dtype = _infer_restore_dtype(
                inputs,
                fallback_dtype=master_dtype,
            )

        self._use_prefetched_layer_forward(layer)
        if self.compute_dtype is None:
            return None
        return _cast_floating(inputs, self.compute_dtype)
```
In the pre-forward hook, we first record the dtype to which floating-point outputs should be restored.
We read the fallback from the persistent local master weight rather than `layer.weight`, because `layer.weight` is replaced by the gathered compute-dtype parameter before the layer runs.
This distinction matters for embeddings: their integer inputs do not supply a floating-point activation dtype, so their outputs fall back to the FP32 master dtype instead of the temporary compute dtype.
We then use the prefetched layer forward function to get the parameters in memory and cast floating-point inputs to the compute dtype.
This concludes the operations we need to do before the forward pass.
But, after the forward pass is over, we need to free the memory of the parameters.
For that, we use the post-forward hook.
```python
    def _post_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...], outputs: torch.Tensor):
        """
        We should restore the local parameters after the forward pass to the layer.
        And clear the full parameter data.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            setattr(layer, param_name, param_state.local_param)
            # Free the forward all-gather buffer. Its storage is shared with the
            # weight autograd saved for backward, so resizing to 0 truly reclaims
            # it; the pre-backward hook re-gathers into the same storage.
            self._free_full_param(param_state.full_param)

        next_index = self._layer_index[layer] + self._prefetch_window_size
        if next_index < len(self._fsdp_layers):
            self._prefetch_layer_forward(self._fsdp_layers[next_index])

        if self.compute_dtype is None:
            return None
        return _cast_floating(outputs, self._layer_states[layer].restore_dtype)
```
We set the layer's parameters to the local parameters and free the full parameter data.
We also prefetch the next layer's parameters so that we can continue the prefetching process.
This is how the overlap between the forward pass and the all-gather operation is achieved.
Also note that, the next index is not necessarily the next layer in the list, but it depends on the prefetch window size.
So, if the prefetch window size is 1, then the next index is the next layer in the list.
If the prefetch window size is 2, then we start prefetching the 2 layers ahead and so on.
Something to note here is that since we are starting the all-gather operation for layer i+1 in the post-forward hook of layer i, the computation of layer i and all-gather of i+1 does not overlap.
But, if we set the window size to 2, then the computation of layer i+1 and all-gather of i+2 does overlap.
We also restore the outputs to the original dtype if we are using the compute dtype.
And this is how we free the memory:
```python
    @staticmethod
    def _free_full_param(full_param: torch.nn.Parameter | None) -> None:
        """Release a full (unsharded) weight by resizing its storage to 0.

        This actually reclaims the memory: autograd's saved-for-backward copy of
        the weight shares this exact storage, so shrinking it frees the bytes for
        real. (Assigning ``data = empty(0)`` does NOT — it just points the param
        at a fresh empty storage and leaves the saved copy holding the old one,
        which is why the forward weights used to stay resident until backward.)
        The tensor keeps its [out, in] sizes so AccumulateGrad still accepts the
        full-shaped gradient."""
        if full_param is None:
            return
        with torch.no_grad():
            full_param.untyped_storage().resize_(0)
```
For this implementation, resizing the existing storage to 0 is necessary because assigning a new empty tensor to `data` would only redirect `full_param`; the tensor saved by autograd would still point to the old underlying memory.
This low-level technique works only because the implementation carefully controls which tensors share that memory and when the hooks run; it should not be read as a generally safe way to mutate arbitrary tensors tracked by autograd.
We then register the forward hooks for the FSDP layers.
```python
    def _register_forward_hooks(self) -> None:
        """
        Register the forward hooks for the FSDP layers.
        """
        for layer in self._fsdp_layers:
            self._forward_hook_handles.append(layer.register_forward_pre_hook(self._pre_forward_hook))
            self._forward_hook_handles.append(layer.register_forward_hook(self._post_forward_hook))
```
One open question might be if we are doing the prefetching for the next index layer, how do we do it for the first layers in the list?
Because if the prefetch window size is 3, then when we run the forward pass for the first layer, we start the prefetching for the 4th layer.
But neither 2nd nor 3rd will have their parameters prefetched when its their turn.
We initialize their prefetching in the forward function directly.
```python
    def forward(self, *inputs, **kwargs):
        """
        Prefetch first FSDP layers params for the forward pass.
        """
        for layer in self._fsdp_layers[:self._prefetch_window_size]:
            self._prefetch_layer_forward(layer)
        outputs = self.module(*inputs, **kwargs)

        for layer in reversed(self._fsdp_layers[-self._prefetch_window_size:]):
            self._prefetch_layer_backward(layer)
        return outputs
```
As we can see, before we run the forward pass of the model (self.module(*inputs, **kwargs)), we prefetch the first few layers.
We will cover the backward pass in the next section, but we do the initial prefetching of the first few layers (which are the last layers in the list) in the backward pass in this code block as well.
Important thing to note here is that the initial prefetching for the backwards layers is done only after the forward pass is over.
So, the mental model is not that we prefetch the last layers for the backward pass at the start of the forward. Its that the forward pass just finished, and the backward pass is imminent, so we prefetch for the first few layers the backward pass will need.

### Backward Pass

#### All-Gather for Backward Pass

We start again by doing the all-gather operation for the parameters of the layer before we do its backward pass.
For a linear layer, `dL/dx = W^T * dL/dy`, so we need the full `W` whenever an input gradient must be computed.
As noted earlier, not every operator or backward path actually needs the parameter values, but this simplified implementation gathers them unconditionally.
```python
    def _regather_full_param_backward(
        self,
        full_param: torch.nn.Parameter,
        local_param: torch.nn.Parameter,
        metadata: ShardMetadata,
    ) -> dist.Work | None:
        """Re-materialize the full weight IN PLACE for backward by refilling the
        same storage the forward pass freed. Because autograd's saved copy shares
        that storage, this is what makes the saved weight valid again (it's the
        backward ALL-GATHER box in the FSDP diagram), rather than allocating a
        second, unused copy."""
        local_shard = local_param.detach()
        if self.compute_dtype is not None:
            local_shard = local_shard.to(self.compute_dtype)

        with torch.no_grad():
            storage = full_param.untyped_storage()
            storage.resize_(metadata.padded_num_elements * full_param.element_size())
            flat = torch.empty(0, dtype=full_param.dtype, device=full_param.device)
            flat.set_(storage, 0, (metadata.padded_num_elements,))
            full_param.data = flat

        if self.world_size == 1:
            with torch.no_grad():
                full_param.data.copy_(local_shard)
            return None
        return dist.all_gather_into_tensor(full_param.data, local_shard, async_op=True)
```
When a layer's backward needs the weight values, autograd saves a tensor that points to the same underlying memory as the full parameter during the forward pass.
Then, with the post-forward hook, we freed the storage of the weight tensor by resizing it to 0.
So the tensor saved by the autograd is now pointing to an empty storage.
To do backward pass, we need to re-materialize the full weight in the same storage so autograd can use it now.
So, in the all-gather for the backward pass, we pick up the storage of the full parameter (which is now pointing to an empty storage) and resize it to the padded number of elements.
Note that when we did the forward pass over the layer, the shape of the parameter was [out, in], and full_param still thinks it is so.
So, we build a flat tensor of the correct size, [padded_num_elements,], and assign it to the data of the full parameter.
Remember that padded_num_elements is the size we need because we have world_size number of GPUs, and each GPU has shard_size number of elements.
And total size is the world_size times the shard_size which is padded_num_elements.
Afterwards, the prefetch is quite similar to the forward pass.
```python
    def _prefetch_layer_backward(self, layer: torch.nn.Module) -> None:
        """
        Issue an async all-gather operation for the layers params for the backward pass.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            full_param = param_state.full_param
            if full_param is None or self._storage_allocated(full_param):
                continue
            metadata = param_state.metadata
            if metadata is None:
                continue
            handle = self._regather_full_param_backward(
                full_param, param_state.local_param, metadata
            )
            param_state.backward_gather_handle = handle
```
We then add it as a backward hook for the layer.
```python
    def _pre_backward_hook(self, layer: torch.nn.Module, grad_output: torch.Tensor):
        """
        Pre-backward hook for the FSDP layers.
        We should gather the parameters before the backward pass.
        """
        self._use_prefetched_layer_backward(layer)

        prev_index = self._layer_index[layer] - self._prefetch_window_size
        if prev_index >= 0:
            self._prefetch_layer_backward(self._fsdp_layers[prev_index])

    def _register_backward_hooks(self) -> None:
        """
        Register the backward hooks for the FSDP layers.
        """
        for layer in self._fsdp_layers:
            self._backward_hook_handles.append(layer.register_full_backward_pre_hook(self._pre_backward_hook))
```
This concludes the all-gather operation for the backward pass.
Now, we need to do the reduce-scatter operation for the gradients.

#### Reduce-Scatter for Backward Pass

Why do we need to do the reduce-scatter operation for the gradients?
Remember that FSDP is data-parallel, meaning each rank uses its own microbatch of data and the gradients are computed on that microbatch.
So, we need to sum the gradients across all the ranks to get the global gradient and this is conceptually the reduce part.
Also, remember that in FSDP, each rank only holds a shard of the parameters, so each rank only needs the gradients for its own shard of the parameters.
So, we need to pass each rank its own gradient and this is conceptually the scatter part.
In the implementation, we first define the data structure for the reduce-scatter operation.
```python
@dataclass
class PendingReduceScatter:
    handle: dist.Work # async Work handle (None when world_size == 1)
    local_grad: torch.Tensor # the local shard grad. valid after the wait()
    local_param: torch.nn.Parameter | None = None # for the finalized grad
```
We store the handle for the reduce-scatter operation so that we can reference to it later.
The local grad is the gradient for the local shard of the parameters and it is valid after the wait() is called on the handle.
The local param is the parameter for which we are computing the gradient and it is used to finalize the gradient.
```python
    def _reduce_scatter_grad_async(self, full_grad: torch.Tensor, metadata: ShardMetadata) -> PendingReduceScatter:
        """
        We issue an async reduce scatter operation.
        """
        flattened_grad = full_grad.flatten()
        if metadata.padded_num_elements > metadata.num_elements:
            padding = torch.zeros(metadata.padded_num_elements - metadata.num_elements, dtype=full_grad.dtype, device=full_grad.device)
            flattened_grad = torch.cat([flattened_grad, padding])

        if self.world_size == 1:
            return PendingReduceScatter(
                handle=None,
                local_grad=flattened_grad,
                local_param=None,
            )
        else:
            local_grad = torch.empty(metadata.shard_size, dtype=full_grad.dtype, device=full_grad.device)
            handle = dist.reduce_scatter_tensor(output=local_grad, input=flattened_grad, op=dist.ReduceOp.SUM, async_op=True)
            return PendingReduceScatter(
                handle=handle,
                local_grad=local_grad,
                local_param=None,
            )
```
The full_grad there is the gradient of the full parameter from a rank.
We first flatten and pad it just like we do to the parameter itself.
Then, we create an empty tensor for the local grad and issue the reduce-scatter operation asynchronously.
Note that the empty tensor is shard_size in size since when we do the reduce-scatter operation, we get the output tensor for the local rank which only owns the shard_size number of elements from the parameter.
```python
    def _make_reduce_scatter_hook(self, layer: torch.nn.Module, param_name: str):
        """
        Return a post-accumulate-grad hook that reduce scatters the full
        weight's gradient into the local shard, then frees the full weight.
        """
        def hook(param: torch.nn.Parameter):
            param_state = self._layer_states[layer].param_states[param_name]
            if param.grad is not None:
                pending_reduce_scatter = self._reduce_scatter_grad_async(param.grad, param_state.metadata)
                # we do pending.local_param is param_state.local_param
                pending_reduce_scatter.local_param = param_state.local_param
                self._pending_reduce_scatters.append(pending_reduce_scatter)
                self._drain_reduce_scatters()
            
            param.grad = None
            # Free the full weight now that this layer's backward has consumed
            # it: resize the storage to 0 (the autograd-saved copy shares it, so
            # the memory is actually reclaimed) rather than detaching to a new
            # empty storage.
            self._free_full_param(param)
        return hook
```
We can now discuss the _make_reduce_scatter_hook that we have attached to the parameter when we were doing the forward pass.
In there, we first issue the async reduce-scatter operation.
For convenience, we also set `pending.local_param = param_state.local_param` so that we can reference the parameter later.
Note that this does not mean replicating the parameter, but just adding a reference to the parameter to the pending reduce scatter object.
Afterwards, we add the pending reduce scatter object to the list of pending reduce scatters and call the _drain_reduce_scatters function.
A question that might arise is are we deleting/freeing the tensors that are needed for reduce-scatter operation before it takes place when we do param.grad = None and self._free_full_param(param)?
The operation has already been enqueued before `param.grad` is cleared; `async_op=True` means that the caller does not synchronously wait for the communication to finish.
With CUDA/NCCL, PyTorch's stream and allocator bookkeeping prevents the input storage from being recycled while the communication stream is still using it.
When padding is needed, `torch.cat` creates a separate padded input buffer, which temporarily increases memory use.
Dropping the last Python reference makes a tensor eligible for deallocation, but its physical storage may not be reusable until the in-flight stream work is complete; calling `wait()` synchronizes use of the result but does not itself free the tensor.
For an implementation that wants to make the lifetime invariant explicit and backend-independent, `PendingReduceScatter` could retain `flattened_grad` until finalization.
Finally, `_free_full_param` frees the full weight's storage, not the gradient's storage.
Now lets look at how we drain the reduce scatters.
```python
    def _finalize_reduce_scatter(self, pending: PendingReduceScatter) -> None:
        if pending.handle is not None:
            pending.handle.wait()

        local_grad = pending.local_grad.to(pending.local_param.dtype).div_(self.world_size)

        if pending.local_param.grad is None:
            pending.local_param.grad = local_grad
        else:
            pending.local_param.grad.add_(local_grad)

    def _drain_reduce_scatters(self) -> None:
        """
        Drain the reduce scatters.
        """
        while len(self._pending_reduce_scatters) > self._reduce_scatter_window_size:
            pending = self._pending_reduce_scatters.pop(0)
            self._finalize_reduce_scatter(pending)
```
Finalizing a reduce scatter simply means waiting for the operation to complete and then dividing the summed gradient by the world size.
We then attach the gradient to the parameter.
Remember that pending.local_param is the parameter_state.local_param.

The key thing to understand here is why we keep a window of pending reduce-scatters instead of finalizing each one immediately.
For a CPU collective, `Work.wait()` blocks the process until completion.
For a CUDA collective, it normally inserts a dependency from the active CUDA stream to the communication stream without blocking the CPU; subsequent GPU work on that stream may still stall until the collective is ready.
If we called `wait()` right after issuing every reduce-scatter, we would place that dependency into the compute stream immediately and lose potential overlap.
Instead, we let a parameter's reduce-scatter run while backward computation for earlier layers proceeds, and only call `wait()` once `_reduce_scatter_window_size` newer reduce-scatters have been queued behind it.
Because this implementation reduce-scatters each parameter separately, this window counts parameter collectives, not layers.
This is the backward-pass analogue of forward prefetching: there we overlap an all-gather with computation, while here we overlap a reduce-scatter with subsequent backward computation.
The window size controls the trade-off: a larger window gives more slack for the collective to progress before the compute stream must depend on it, at the cost of holding more in-flight gradient buffers in memory.

Because we keep this window, some reduce-scatters are still in flight when the backward pass ends.
So before the optimizer step we call `finish_gradient_synchronization` to drain every remaining pending reduce-scatter and only then sync the replicated parameters.

The last part is the replicated parameters part.
```python
    def _sync_grads_of_replicated_parameters(self) -> None:
        """
        Sync the gradients of the replicated parameters.
        The replicated parameters are not sharded, so we can just all-reduce the gradients.
        """
        if self.world_size == 1:
            return
        for param in self._replicate_parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad.div_(self.world_size)
```
Since the replicated parameters are not sharded, we can just all-reduce the gradients.
Additionally, we do
```python
    def finish_gradient_synchronization(self):
        """
        Wait for all the reduce scatter operations to complete.
        """
        while self._pending_reduce_scatters:
            pending = self._pending_reduce_scatters.pop(0)
            self._finalize_reduce_scatter(pending)

        self._sync_grads_of_replicated_parameters()
```
This is called once before the optimizer step.
