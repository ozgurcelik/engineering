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
This means if the model and gradients are in BF16, then in baseline we would have (2 + 2 + K) * N_params bytes of memory per GPU, where N_params is the number of parameters in the model and K is the optimizer state overhead per parameter.
With FSDP, we can reduce this to (2 + 2 + K) * N_params / N_gpus bytes of memory per GPU, where N_gpus is the number of GPUs in the world.

## How does FSDP work?

As we run the FSDP, each GPU will hold a shard of the model parameters.
But to be able to do the forward and backward pass through a layer in a GPU, we will need all the parameters for a layer materialized in the GPU.
This requires us to do an all-gather operation before the forward pass.
Once the forward pass is done, we can free the memory of the parameters.
This means, during the forward pass, we can get away with fully holding the parameters of only one layer at a time.
During the backward pass, we will need all the parameters for the layer that we are currently processing once again.
Why is that?
Imagine we have a linear layer with parameters `W` and input `x` generating output `y`.
The forward pass is given by `y = Wx`.
And we have the loss function `L` with `dL/dy` available to us.
Then, the gradients for this layer is given by `dL/dW = dL/dy * x^T` and this does not need us to have the full `W` in memory.
But, the input gradients `dL/dx = W^T * dL/dy` does need us to have the full `W` in memory.
This means, before we can the backward pass over a layer, we will need to do another all-gather operation to get the full `W` in memory.
And after doing the backward pass, we can free the memory of the parameters just like we did in the forward pass.
Once we have computed the gradients for a layer, we need to do reduce-scatter operation so that each GPU can get the gradients for its shard of the model.
So, in total, we will do 2 all-gathers and 1 reduce-scatter operations leading to 3 * N_params communication cost.
In data parallel, we would have done 1 all-gather and 1 reduce-scatter operations leading to 2 * N_params communication cost, so FSDP has 1.5 times the communication cost of data parallel.

## The implementation

### Initializations

First of all, we need to find all the layers that are going to be sharded.
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
Start and end indices are the indices of the first and last element in the shard for this rank.
These two will have different values for each rank.

Now, we need to create a `FSDPLayerState` object for each layer that is going to be sharded.
```python
    def _create_layer_state(self, layer: torch.nn.Module) -> FSDPLayerState:
        """
        Create a FSDPLayerState object for the layer.
        """
        return FSDPLayerState(layer=layer)
```
In the ParamState object, we store the metadata for the parameter, the local parameter which is the shard of the parameter for this rank, and the full parameter which will be rematerialized later for the forward and backward passes and freed when we are done with it.
We also store the handles for the forward and backward gather operations which are done asynchronously.

In the LayerState object, we have the param_states dictionary to store the ParamState objects for each parameter in the layer.
Remember that linear layer has weight and bias parameters each with their own ParamState object.
Restore dtype is so that we can remember the dtype of the parameters of the layer.

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
So, the parameters we use in the forward and backward pass computation might be lower precision, but we accumulate the gradients on the master weight which is in float32.
Also, we set the layer's parameters to the local parameter instead of the full parameter.

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
In the prefecth function, we iterate over all the parameters of the layer (such as weight and bias of a linear layer) and check if the full parameter is already allocated.
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
        self._use_prefetched_layer_forward(layer)
        if self.compute_dtype is None:
            return None
        self._layer_states[layer].restore_dtype = _infer_restore_dtype(layer, inputs)
        return _cast_floating(inputs, self.compute_dtype)
```
In the pre-forward hook, we use the prefetched layer forward function to get the parameters in memory.
If we are using the compute dtype, we also cast the inputs to the compute dtype.
We also remember the restore dtype so that we can restore the inputs to the original dtype after the forward pass.
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
Resizing the storage to 0 is more robust than setting the data to an empty tensor.
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
We initialize their prefecting in the forward function directly.
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
So, the mental model is not that the prefecth last layers for the backward pass at the start of the forward. Its forward pass just finished, and backwards pass is imminent, so prefetch for the first few layers backwards will need.

### Backward Pass

#### All-Gather for Backward Pass

We start again by doing the all-gather operation for the parameters of the layer before we can do the backward pass.
Remember that `dL/dx = W^T * dL/dy` so we need the full `W` in memory to compute the input gradients.
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
In the forward pass, when we run this layer, autograd saved the weight tensor for the backward pass.
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
For ease, we also do the pending.local_param is param_state.local_param so that we can reference to the parameter later.
Note that this does not mean replicating the parameter, but just adding a reference to the parameter to the pending reduce scatter object.
Afterwards, we add the pending reduce scatter object to the list of pending reduce scatters and call the _drain_reduce_scatters function.
A question that might arise is are we deleting/freeing the tensors that are needed for reduce-scatter operation before it takes place when we do param.grad = None and self._free_full_param(param)?
The answer is no, the operation is safe.
There are a few things we need to understand here:
1. Async op does not mean that the operation is deferred.
The operation is enqueued on hte nccl stream immediately while the param.grad still has the valid data.
Async there means that cpu thread does not block waiting for completion of the operation and gpu-side read of flattened grad is already scheduled.
2. Setting the param.grad to None drops the reference to the tensor but the data stays alive as long as something references to it.
3. The work handle still references to the tensor until the operation is completed.
Additionaly, the padding allocates a new tensor so for the padding branch, the param.grad part is more obvious.
And in that case, the param.grad does actually free the memory there and then since the referenced tensor is now the padded one.
This also means padding momentarily increases the memory usage until param.grad is set to None.
And the padded tensor's memory is freed when we wait for the handle to complete.
And, the _free_full_param frees the weight, and not the grad.
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
before the optimizer step where we finish any hanging reduce scatter operations and sync the gradients of the replicated parameters.