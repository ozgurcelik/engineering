# FSDP From Scratch

## Introduction

The purpose of this project is to learn how FSDP works and implement it from scratch.

## Why FSDP?

FSDP stands for Fully Sharded Data Parallel. It is still data parallelism: every rank has the same logical model but processes a different slice of the input batch.

In ordinary data parallelism, every rank keeps a complete copy of the parameters, gradients, and optimizer state. That replication becomes the limiting factor when the model state no longer fits on one device. FSDP instead divides the persistent model state across ranks. Each rank owns only a shard of a sharded parameter, the corresponding gradient shard, and the optimizer state for that shard.

Sharding persistent state does not mean that a rank never holds a full parameter. A layer still needs a usable full parameter while it computes. FSDP therefore materializes full parameters temporarily, one layer or prefetch window at a time, and releases them afterward. The memory saving comes from what remains resident between computations.

In our implementation, we will look at a simplified version of FSDP where we assume a strictly linear, single-use execution order: every sharded module is entered exactly once in a forward graph, and only one such graph is outstanding when backward begins.
This keeps the lifecycle of each parameter easy to reason about, at the cost of ruling out several patterns (shared/tied parameters, modules invoked more than once, activation checkpointing, and clean frozen-parameter memory handling). We defer the details to the [Limitations](#limitations) section at the end, once the mechanism is in place.
This is therefore a learning implementation, not a production-ready FSDP replacement. It also flattens and shards each parameter separately.
By contrast, [PyTorch FSDP1](https://docs.pytorch.org/docs/stable/fsdp.html) concatenates the parameters managed by an FSDP unit into a `FlatParameter`.
[FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) also represents shards per parameter, but it chunks each parameter along dimension 0 and represents the result as a `DTensor`; our implementation instead flattens the entire parameter before slicing it.
FSDP2 still groups multiple parameters so that each group uses one all-gather and one reduce-scatter.


## How does FSDP work?

In this implementation, linear and embedding parameters are sharded. Smaller parameters, such as normalization weights, remain replicated in full on every rank. Two collective operations move the sharded values between their persistent and temporary forms:

- **All-gather:** every rank contributes its local parameter shard, and every rank receives the concatenation of all shards. This reconstructs the full parameter needed for computation.
- **Reduce-scatter:** every rank contributes a full gradient computed from its local microbatch. The gradients are summed across ranks, split into shards, and each rank receives only the reduced shard that corresponds to its local parameter shard.

### A concrete sharding example

Suppose a flattened parameter contains 10 elements and the world size is 3. The shard size is `ceil(10 / 3) = 4`, so we pad the parameter to 12 elements:

```text
full padded parameter: [w0, w1, w2, w3 | w4, w5, w6, w7 | w8, w9, 0, 0]
rank 0 local shard:    [w0, w1, w2, w3]
rank 1 local shard:    [w4, w5, w6, w7]
rank 2 local shard:    [w8, w9, 0,  0 ]
```

An all-gather concatenates these shards in rank order. Every rank receives the same 12-element buffer, discards the final two padding elements, and reshapes the first 10 elements to the parameter's original shape.

During backward, let `g[r][i]` denote rank `r`'s gradient for element `i`. Each rank pads its full 10-element gradient to 12 elements and splits it into the same three chunks. Reduce-scatter sums corresponding chunks across ranks and sends chunk `r` to rank `r`. Rank 1 receives:

```text
[sum_r g[r][4], sum_r g[r][5], sum_r g[r][6], sum_r g[r][7]]
```

The implementation divides the result by 3 to average equally weighted rank-local gradients, converts it to FP32, and attaches it to rank 1's persistent `[w4, w5, w6, w7]` shard. Padding remains part of rank 2's local shard, but it starts at zero and always receives a zero gradient. It therefore stays zero for the optimizers considered here, although the optimizer may still allocate state and perform wasted elementwise work for those slots.

### Why gather again for backward?

Before a layer's forward computation, an all-gather reconstructs its parameters. After the layer finishes, those full parameters can be released. Without prefetching, only one layer's full parameters need to be materialized at a time. Prefetching deliberately holds additional upcoming layers so communication can overlap computation, trading a higher transient memory peak for throughput.

Backward may need the full parameter again. Consider a linear layer with weight `W` of shape `[out, in]`, input `x`, and output `y`. We will reason with column-vector inputs, so `y = W x`; PyTorch's `nn.Linear` computes the equivalent `y = x W^T` for a batch of row vectors, but the gradient structure is identical.
Given the loss `L` and the upstream gradient `dL/dy`, the two gradients this layer produces are:

- the weight gradient `dL/dW = dL/dy · x^T`, which needs `dL/dy` and `x` but *not* `W`; and
- the input gradient `dL/dx = W^T · dL/dy`, which *does* need the full `W`.

The weight gradient does not require `W`, but the input gradient does. Some paths—such as embedding backward, or a first linear layer whose input does not require a gradient—do not need the parameter value at all. This toy stays operator-agnostic: whenever a sharded module participates in backward, its module hook regathers all of that module's parameters. After backward consumes the layer, the full storage can be released again.

### One Training Iteration at a Glance

The values of one logical parameter move through the following lifecycle. The local FP32 shard remains the persistent source of truth; the full compute parameter is a temporary object created from all ranks' shards.

```text
persistent local FP32 shard
        |
        +-- forward all-gather (optionally cast to compute_dtype)
        |       -> temporary full parameter
        |       -> forward computation
        |       -> release full storage
        |
        +-- backward all-gather from the same local shard
                -> refill the temporary full-parameter storage
                -> backward computation
                -> rank-local full gradient
                -> reduce-scatter + divide by world size
                -> reduced local FP32 gradient
                -> optimizer updates the persistent local shard
```

Forward visits layers in model order, while backward normally visits them in reverse order. `finish_gradient_synchronization()` waits for any reduce-scatters still in flight and synchronizes replicated gradients before the optimizer updates the persistent local shards. Replicated parameters skip this lifecycle: they stay resident in FP32 on every rank, and their gradients are all-reduced inside `finish_gradient_synchronization()`.

### Using the wrapper

The distributed process group must already be initialized, and the model must already be on the rank-local device, before constructing `FSDP`. The wrapper reads its rank and world size once during construction.

After that setup, the caller-facing API is small: wrap the model, build the optimizer over the *wrapped* module's parameters, and call `finish_gradient_synchronization()` after `backward()` and before `optimizer.step()`:

```python
fsdp_model = FSDP(model, compute_dtype=torch.bfloat16)

# IMPORTANT: build the optimizer AFTER wrapping.
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=3e-4)

for input_ids, labels in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(fsdp_model(input_ids), labels)
    loss.backward()
    fsdp_model.finish_gradient_synchronization()  # drain reduce-scatters + sync replicated grads
    optimizer.step()
```

Two details are load-bearing. First, the optimizer must be created *after* wrapping: `_create_layer_state` replaces each sharded layer's `weight` and `bias` with freshly created local shards. An optimizer built beforehand would still refer to the now-orphaned full parameters and would silently update the wrong tensors. `fsdp_model.parameters()` returns the intended set: the per-rank FP32 local shards plus the replicated parameters. Second, `finish_gradient_synchronization()` must run before every `optimizer.step()`, because the reduce-scatter window intentionally leaves some gradient collectives in flight when `backward()` returns.

## The implementation

We will follow the runtime lifecycle in order: construction creates persistent shards, forward materializes and releases full parameters, backward rematerializes them and reduce-scatters gradients, and finalization prepares those local gradients for the optimizer.

### Initializations

Construction has three jobs: synchronize the initial model, decide which parameters to shard, and replace those parameters with persistent local shards.

First, rank 0 broadcasts the initial parameters and buffers so every rank starts from identical values. Next, the wrapper finds the modules it will shard. This toy shards `Linear` and `Embedding` parameters; every other trainable parameter remains replicated and its gradient is later all-reduced. The replicated set typically contains normalization parameters, which are small enough that sharding them would add communication without saving much memory.

The lists `_fsdp_layers` and `_replicate_parameters` record those two groups:
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
Doing this at the module level keeps the logic simple: a `Linear` or `Embedding` is short-circuited into `_fsdp_layers` as a whole (so its `weight` and `bias` are handled together later), while every other module contributes only its own parameters.
Because `module.modules()` also visits parent containers, `recurse=False` is what stops a parent from re-counting a child's parameters.

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

There is a subtlety worth calling out here. Because we cast the model's inputs to `compute_dtype` once and then let activations flow in that dtype, a replicated FP32 norm receives a low-precision activation while holding an FP32 weight. This only works if the norm itself handles the mixed dtypes: a well-behaved `RMSNorm` upcasts the activation to FP32 internally and casts the result back to the incoming dtype, and `nn.LayerNorm` promotes internally as well. A naive norm that assumed its input and weight share a dtype could hit a dtype-mismatch error on this path.

The wrapper also needs enough metadata to move each parameter between its local shard and full shape. `FSDPLayerState` owns the state for one sharded module, while `FSDPParamState` owns the state for one of that module's parameters.

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
```
`ShardMetadata` records the relationship between the original parameter and its padded, flattened shards. The following helper computes that layout:
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
`num_elements` is the parameter's actual element count. Because that count may not be divisible by the world size, `padded_num_elements` rounds it up so every rank owns an equally sized shard. `shape` preserves the original shape for reconstruction.
`start` is the index of the first element in this rank's shard, while `end` is the exclusive endpoint of the half-open slice `[start, end)`.

`FSDPParamState` stores this metadata, the persistent local parameter shard, the temporary full parameter, and the handles for asynchronous forward and backward all-gathers. A linear layer therefore has one parameter state for its weight and, when present, another for its bias.

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

The wrapper flattens and pads each parameter, takes this rank's slice, converts it to FP32, and installs that shard on the layer. When `compute_dtype` is configured, later all-gathers cast these FP32 master shards to the lower precision used for computation. Autograd produces full gradients and reduce-scatter communicates them in that lower dtype; finalization converts each reduced shard back to FP32 for the optimizer.

This local parameter is what closes the training loop.
The optimizer steps on the FP32 local shards (`local_param`), never on the full or compute-dtype parameters.
Each rank owns a disjoint slice of the real parameter elements, so optimizer updates for those elements are not duplicated across ranks. Padded elements are the exception: they still incur wasted optimizer state and elementwise work.
This is exactly why the optimizer states (for example Adam's moments) end up sharded too: they are created and kept per `local_param`, so each rank only stores the optimizer state for its own shard.
After the step, the updated FP32 values live in `local_param`, and they are re-cast to the compute dtype the next time we all-gather that layer, so the following forward pass automatically sees the freshly updated weights.

### Forward Pass

The forward path repeats three steps for every sharded layer:

1. issue an asynchronous all-gather early enough to overlap it with other computation;
2. wait for the gather when the layer is about to run, then install the full parameters; and
3. after the layer finishes, restore the persistent local shards and release the full-parameter storage.

The first building block issues the asynchronous all-gather:
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
With one rank, no communication is necessary. Otherwise, the function optionally casts the local FP32 shard to `compute_dtype`, allocates the full padded output buffer, and launches the collective. The buffer must not be read until the returned handle has completed.
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
`_prefetch_layer_forward` does this for each parameter in a layer unless a full parameter is already materialized. It immediately wraps the output buffer in a `Parameter` and, for trainable parameters, registers the reduce-scatter hook discussed in the backward section. The hook must be attached before forward because autograd records the exact `Parameter` object that participates in the computation. The gather handle is saved so the pre-forward hook can wait for it later.
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
When the layer is ready to run, `_use_prefetched_layer_forward` waits for its gather, removes padding, restores the original shape, and installs the full `Parameter` on the layer. If prefetching has hidden the communication successfully, the wait returns immediately. Only the parameter's `.data` view changes, so the previously registered gradient hook remains attached to the same object.
```python
    def _pre_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        """
        Pre-forward hook for the FSDP layers.
        Wait for the prefetched parameters before the layer's forward pass.
        """
        self._prefetch_layer_forward(layer) # if the layer is already prefetched, this is a no-op
        self._use_prefetched_layer_forward(layer)
```
The pre-forward hook first calls the prefetch function as a just-in-time fallback. In the normal prefetched case that call is a no-op. It then waits for the gather and installs the materialized parameters. Activation casting happens once at the outer `FSDP.forward` boundary, not at each layer.

The post-forward hook performs the inverse transition:
```python
    def _post_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...], outputs: torch.Tensor):
        """
        We should restore the local parameters after the forward pass to the layer.
        And clear the full parameter data.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            setattr(layer, param_name, param_state.local_param)
            # Free the forward all-gather buffer. Its storage is shared with the
            # weight autograd saved for backward, so resizing to 0 releases that
            # shared allocation; the pre-backward hook re-gathers into the same
            # storage.
            self._free_full_param(param_state.full_param)

        next_index = self._layer_index[layer] + self._prefetch_window_size
        if next_index < len(self._fsdp_layers):
            self._prefetch_layer_forward(self._fsdp_layers[next_index])
```
It restores each layer attribute to its persistent local shard, releases the temporary full storage, and launches a gather for the layer `window` positions ahead. The window is a lookahead distance: layer `i` launches the gather for layer `i + window`. With a window of 1, `AG(L{i+1})` starts only after `L{i}` finishes and has no later sharded-layer computation to overlap. With a window of 2, `AG(L{i+2})` can overlap the computation of `L{i+1}`.

The prefetch window is therefore both a lookahead distance and a transient-memory trade-off. With a window size of 2, the forward schedule looks like this:

| Moment | Action | Intended overlap |
| --- | --- | --- |
| Before the model starts | Issue `AG(L0)` and `AG(L1)` | Seed the first two layers |
| Pre-forward of `L0` | Wait for `AG(L0)`, then compute `L0` | `L1` is already gathered or in flight |
| Post-forward of `L0` | Release `L0`; issue `AG(L2)` | `AG(L2)` can overlap with computation of `L1` |
| Post-forward of `L1` | Release `L1`; issue `AG(L3)` | `AG(L3)` can overlap with computation of `L2` |

```text
forward order:       L0          L1          L2          L3
compute:           [ F0 ]      [ F1 ]      [ F2 ]      [ F3 ]
prefetched gather:  AG0, AG1       [ AG2 ]     [ AG3 ]
                                  ^ issued     ^ issued
                                    after F0     after F1
```

So a larger window creates more opportunity for overlap, but holds more full or in-flight parameter buffers at once.

This code does not create CUDA streams itself. It relies on `async_op=True` and delays `wait()`. On CUDA/NCCL, the collective uses a communication stream and `Work.wait()` makes the active compute stream depend on it; on CPU/Gloo, the asynchronous work progresses in the background. Production FSDP performs more deliberate stream and memory scheduling.

Backward mirrors the schedule in reverse. With four layers and a window of 2, `L3` and `L2` are gathered just in time when backward first reaches them. The pre-backward hook for `L3` starts `AG(L1)`, and the hook for `L2` starts `AG(L0)`, allowing those gathers to overlap the backward computation of later layers.

#### A toy-specific storage trick

The collectives above are fundamental to FSDP; the following storage-resizing mechanism is not. It is a deliberately low-level technique this toy uses to release a full parameter after forward while preserving the tensor objects already captured by autograd.

The helper that releases the storage is:
```python
    @staticmethod
    def _free_full_param(full_param: torch.nn.Parameter | None) -> None:
        """Release a full (unsharded) weight by resizing its storage to 0.

        Autograd's saved-for-backward copy of the weight shares this exact
        storage, so shrinking it releases the shared allocation for reuse by the
        allocator. Assigning ``data = empty(0)`` does not: it points the parameter
        at a fresh empty storage while the saved copy keeps the old allocation
        alive. On CUDA, releasing a live allocation for reuse does not necessarily
        reduce the amount of memory reserved by the caching allocator. The tensor
        keeps its [out, in] sizes so AccumulateGrad still accepts the full-shaped
        gradient."""
        if full_param is None:
            return
        with torch.no_grad():
            full_param.untyped_storage().resize_(0)
```
The toy uses this trick in three places: after forward, before backward, and after the backward reduce-scatter.

A materialized full parameter is just a tensor whose `[out, in]` view points into an all-gather buffer's storage. When autograd needs the weight for backward, it saves a tensor that shares *that exact storage*. Three things therefore point at one allocation:

```text
one allocation (untyped storage)
   ^              ^                     ^
   |              |                     |
full_param        full_param.data       autograd-saved tensor
(the Parameter)   ([out, in] view)      (shares the SAME storage)

free     = storage.resize_(0)        -> every view now points at empty storage
regather = storage.resize_(n*bytes)  -> refills in place; the saved tensor is valid again
```

So to actually free the weight we must shrink the storage itself, not merely repoint `full_param`: assigning a fresh empty tensor to `full_param.data` would leave the autograd-saved copy holding the old allocation alive. Resizing the storage to 0 releases the shared allocation for reuse while keeping the tensor's `[out, in]` sizes intact, so `AccumulateGrad` still accepts a full-shaped gradient later. This is safe only because we control exactly which tensors share the storage and when the hooks run; it is not a general-purpose way to mutate autograd-tracked tensors.

The mirror image of "free by resizing to 0" is the question "is this parameter currently materialized?", which we answer by asking the storage, not the tensor shape:
```python
    @staticmethod
    def _storage_allocated(param: torch.nn.Parameter | None) -> bool:
        """A full param's memory is tracked by its STORAGE, not its tensor
        sizes: we free by resizing the storage to 0 while keeping the [out, in]
        sizes (so autograd still sees the right shape). So 'is it materialized?'
        must ask the storage, not `.data.numel()`."""
        return param is not None and param.untyped_storage().size() > 0
```
A freed full parameter still reports its logical `[out, in]` shape, so `.numel()` says nothing about whether its backing allocation is present. The materialization guards must therefore inspect `untyped_storage().size()`.

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

The post-forward chain cannot seed itself: with a window of 3, layer 0 launches the gather for layer 3, leaving layers 0 through 2 without an earlier hook to prefetch them. `FSDP.forward` therefore launches gathers for the first `window` layers before entering the model. Backward has the analogous startup problem, but its pre-backward hook handles it just in time.
```python
    def forward(self, *inputs, **kwargs):
        """
        Prefetch the first FSDP layers and cast model inputs once for mixed
        precision. Activations remain in compute_dtype unless the model itself
        explicitly changes their dtype.
        """
        for layer in self._fsdp_layers[:self._prefetch_window_size]:
            self._prefetch_layer_forward(layer)

        if self.compute_dtype is not None:
            inputs = _cast_floating(inputs, self.compute_dtype)
            kwargs = _cast_floating(kwargs, self.compute_dtype)

        outputs = self.module(*inputs, **kwargs)

        return outputs
```
Before entering the model, `forward` prefetches the first few layers and casts floating-point model inputs to `compute_dtype` once.
The helper recursively handles tensors in positional arguments, lists, tuples, and keyword-argument dictionaries, while leaving integer and boolean tensors such as embedding indices unchanged.
We do not cast each sharded layer's output back to its incoming dtype, so activations normally remain in `compute_dtype` as they flow through the model.
This avoids repeated FP32-to-low-precision-to-FP32 conversions and is closer to the usual module-level FSDP mixed-precision policy.

### Backward Pass

#### All-Gather for Backward Pass

Backward starts by rematerializing the current layer's parameters. For a linear layer, `dL/dx = W^T * dL/dy`, so the full `W` is required whenever an input gradient must be computed. Not every operator or backward path needs parameter values, but this simplified module-level policy gathers every parameter belonging to a sharded module whose backward hook runs.
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
This is the backward half of the toy-specific storage trick. The post-forward hook left the existing tensor objects and their shape metadata intact but shrank their shared storage to zero. `_regather_full_param_backward` grows that same storage, temporarily exposes it as a flat padded buffer, and all-gathers into it. `_use_prefetched_layer_backward` later removes the padding and restores the original-shaped view. Reusing the storage matters when autograd saved a tensor that shares it, as happens for a linear weight needed to compute an input gradient.

Backward prefetching then follows the same issue-now, wait-later pattern as forward prefetching:
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
The module backward pre-hook consumes this prefetch operation:
```python
    def _pre_backward_hook(
        self,
        layer: torch.nn.Module,
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        """
        Pre-backward hook for the FSDP layers.
        We should gather the parameters before the backward pass.
        """
        # Self-heal the first `window` layers of the backward pass: no later
        # layer prefetched them, so gather them just-in-time here. For layers
        # already prefetched by the chain below, this is a no-op because the
        # storage is already allocated.
        self._prefetch_layer_backward(layer)
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
The hook performs three ordered actions:

1. It calls `_prefetch_layer_backward(layer)` for the current layer. For the first `window` layers encountered in backward—the last layers from forward—this seeds the schedule just in time. For later layers the gather is already in flight, so the call is a no-op.
2. It waits for the current gather and restores the original-shaped full storage. It does not reattach the full parameter to the module: backward follows the parameter and saved tensors already captured by the forward autograd graph.
3. It launches the gather for the layer `window` positions earlier in forward order, allowing that communication to overlap the current layer's backward computation.

Let us look at `_use_prefetched_layer_backward` more closely, since it is what turns the flat, padded buffer from the all-gather back into a usable weight.
```python
    def _use_prefetched_layer_backward(self, layer: torch.nn.Module) -> None:
        """
        Wait for the all-gathers, trim the padding, and reshape the full
        parameter data used by the existing autograd graph.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if param_state.backward_gather_handle is not None:
                param_state.backward_gather_handle.wait()
            param_state.full_param.data = param_state.full_param.data[:param_state.metadata.num_elements].view(param_state.metadata.shape)
            param_state.backward_gather_handle = None
```
This helper waits for the all-gather, trims padding, restores the full parameter's original shape, and clears the handle. No `setattr` is needed because autograd already holds the relevant full-parameter references from forward.

#### Reduce-Scatter for Backward Pass

Each rank now has a full gradient derived from its own microbatch, but its optimizer owns only a local parameter shard. Reduce-scatter combines the required synchronization and sharding: it sums full gradients across ranks and returns only the corresponding gradient shard to each rank.

The following structure tracks one asynchronous reduce-scatter until its result can be attached to the persistent local parameter:
```python
@dataclass
class PendingReduceScatter:
    handle: dist.Work | None  # async Work handle (None when world_size == 1)
    local_grad: torch.Tensor  # the local shard grad; valid after wait()
    local_param: torch.nn.Parameter | None = None  # receives the finalized grad
```
`handle` tracks the asynchronous collective, `local_grad` is its output shard, and `local_param` identifies the persistent parameter that will receive the finalized gradient.
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
`full_grad` is this rank's gradient for the full parameter. The function flattens and pads it using the same layout as the parameter, allocates a `shard_size` output, and launches reduce-scatter asynchronously.
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
                # share the same object (a reference, not a copy)
                pending_reduce_scatter.local_param = param_state.local_param
                self._pending_reduce_scatters.append(pending_reduce_scatter)
                self._drain_reduce_scatters()
            
            param.grad = None
            # Free the full weight now that this layer's backward has consumed
            # it: resize the storage to 0 (the autograd-saved copy shares it, so
            # the shared allocation is released for reuse) rather than detaching
            # to a new empty storage.
            self._free_full_param(param)
        return hook
```
The post-accumulate-grad hook starts that collective as soon as the full gradient is ready. It records which persistent local parameter should receive the result, appends the work to the pending queue, clears the temporary full gradient, and releases the full parameter's storage.

Clearing `param.grad` and the full weight does not clear the reduce-scatter output stored in `PendingReduceScatter`. The collective also needs its flattened input until the backend has consumed it; this toy relies on PyTorch's asynchronous collective machinery for that lifetime. A more defensive teaching implementation could store the flattened input in `PendingReduceScatter` until finalization. Padding with `torch.cat` also creates an extra temporary input allocation.

The pending queue is drained as follows:
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
Finalization waits for the collective, divides the summed shard by the world size, converts it to the persistent parameter's FP32 dtype, and either installs or accumulates the result in `local_param.grad`. This is the gradient of a global per-example mean when ranks contribute equally weighted local batches; uneven batch sizes require weighted reduction.

The key thing to understand here is why we keep a window of pending reduce-scatters instead of finalizing each one immediately.
For a CPU collective, `Work.wait()` blocks the process until completion.
For a CUDA collective, it normally inserts a dependency from the active CUDA stream to the communication stream without blocking the CPU; subsequent GPU work on that stream may still stall until the collective is ready.
If we called `wait()` right after issuing every reduce-scatter, we would place that dependency into the compute stream immediately and lose potential overlap.
Instead, we let a parameter's reduce-scatter run while backward computation for earlier layers proceeds, and only call `wait()` once `_reduce_scatter_window_size` newer reduce-scatters have been queued behind it.
This is the backward-pass analogue of forward prefetching: there we overlap an all-gather with computation, while here we overlap a reduce-scatter with subsequent backward computation.
The window size controls the trade-off: a larger window gives more slack for the collective to progress before the compute stream must depend on it, at the cost of holding more in-flight gradient buffers in memory.
Concretely, after each new reduce-scatter is appended, `_drain_reduce_scatters` finalizes the oldest operations until at most `_reduce_scatter_window_size` remain un-waited. With a window of 2, for example, the first operation is finalized when the third is appended, at which point two newer operations are queued behind it.
> **Note — the two window sizes are not in the same units.** Because this implementation reduce-scatters each parameter separately, the reduce-scatter window counts individual parameter collectives (a layer's weight and bias reduce-scatter independently), whereas the forward and backward all-gather windows are counted in whole layers.

Because the window deliberately leaves recent operations pending, some reduce-scatters may still be in flight when `backward()` returns. `finish_gradient_synchronization()` drains them before the optimizer step.

Replicated parameters do not need reduce-scatter because every rank owns their full shape. Their gradients use an ordinary all-reduce:
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
Finally, `finish_gradient_synchronization()` combines both synchronization paths:
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
The caller invokes this once after backward and before the optimizer step.

## Costs and trade-offs

The sharding lifecycle is easier to understand first; we can now quantify its main trade-off.

For the sharded parameters in this toy, an AdamW training step keeps approximately four FP32 values per real parameter element across the persistent parameter, finalized gradient, and two optimizer moments. That is about 16 bytes per element in ordinary data parallelism and `16 / P` bytes per rank with FSDP over `P` ranks, ignoring padding. The lower-precision full parameters and full gradients are temporary additions while a layer or prefetch window is active.

Communication increases in exchange for that memory saving. For `N_params` trainable sharded elements and world size `P`, this implementation performs two all-gathers and one reduce-scatter per iteration. Ignoring padding and latency, the conventional one-direction per-rank bandwidth term is approximately:

```text
FSDP: 3 * (P - 1) / P * N_params elements
DDP:  2 * (P - 1) / P * N_params elements
```

The DDP term comes from a bandwidth-optimal all-reduce, which is equivalent to a reduce-scatter followed by an all-gather. Under these assumptions, FSDP communicates 1.5 times as many parameter elements as DDP. The estimate excludes activations, transient prefetch buffers, padding, replicated-parameter all-reduces, allocator overhead, and the latency cost of issuing one collective per parameter.

## Limitations

This implementation is deliberately simplified. It assumes a strictly linear, single-use execution order — every sharded module is entered exactly once per forward graph, and only one such graph is outstanding when backward begins. That assumption rules out several patterns:

- **Shared / tied parameters.** If two modules share a parameter — for example a tied input embedding and output projection — wrapping replaces each module's attribute with a *separately created* local shard, silently breaking the tie.
- **Modules invoked more than once per graph (or two forwards before a backward).** Each parameter has a single `FSDPParamState`, which can only remember the most recently created `full_param` and gather handles. During backward, an earlier invocation may still refer to a different full parameter whose storage has already been freed.
- **Activation checkpointing.** Recomputation re-invokes the forward hooks *during* backward, which would require checkpoint-aware scheduling and per-invocation state.

Supporting these cases needs alias-aware parameter ownership plus a stack (or equivalent per-invocation lifecycle record); a simple reference count is not enough. Ordinary gradient accumulation (repeated `forward -> backward` pairs before a single `optimizer.step()`) *is* compatible with this lifecycle, though this implementation still communicates on every backward.

**Frozen parameters** are numerically supported — the wrapper preserves `requires_grad=False`, they receive no gradient, and the optimizer does not update them — but their post-backward memory handling is incomplete. The post-forward hook releases every materialized full parameter; if backward later needs a frozen parameter's value, the pre-backward hook re-gathers it, and because the releasing post-accumulate-grad hook is only registered for *trainable* parameters, the re-gathered full storage stays allocated after backward. A frozen parameter on a path that never needs its value in backward (such as some frozen embeddings) is simply never re-gathered and avoids this issue.

Finally, this is a learning implementation, not a production one: we flatten and shard each parameter separately and issue a collective per parameter, which adds latency and padding overhead compared to grouped implementations like PyTorch FSDP1/FSDP2.

## Why the storage-resizing trick is needed

The storage-resizing code is not a fundamental part of FSDP. It is a consequence of combining ordinary PyTorch autograd with our decision to free a gathered parameter after forward and gather it again for backward.

During forward, autograd may save a tensor that shares the full parameter's storage. For example, linear backward needs the full weight to compute the input gradient. Restoring `layer.weight` to the local shard does not change the tensor already saved by autograd, so that saved tensor would continue to keep the full allocation alive. Assigning `full_param.data` to a new empty tensor would not solve this either: it would only repoint `full_param`, while autograd's saved tensor would still reference the old storage.

We therefore shrink the shared storage itself to zero after forward. This releases the full allocation while leaving the existing tensor objects, shapes, and autograd graph intact. Before backward, we grow that same storage and all-gather the weight into it again. Because autograd's saved tensor still refers to that storage, it sees the newly gathered values and backward can proceed. After the full gradient has been reduce-scattered, we shrink the storage once more.

We could avoid this low-level technique by retaining every gathered weight until backward, but then the full parameters would accumulate throughout forward and largely defeat reshard-after-forward memory savings. A more robust alternative would be custom autograd logic that explicitly regathers parameters during backward, but that would be more operator-specific and substantially more complex. This implementation uses storage resizing as a compact teaching mechanism; mutating `.data` and raw storage this way is fragile and should not be treated as a general PyTorch pattern.
