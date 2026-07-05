from __future__ import annotations

import torch
import torch.distributed as dist

from dataclasses import dataclass, field


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

@dataclass
class FSDPLayerState:
    layer: torch.nn.Module
    param_states: dict[str, FSDPParamState] = field(default_factory=dict)

class FSDP(torch.nn.Module):
    def __init__(self, 
                 module: torch.nn.Module,
                 compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.module = module
        self.compute_dtype = compute_dtype

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self._broadcast_initial_state()

        self._layer_states: dict[torch.nn.Module, FSDPLayerState] = {}
        self._forward_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._backward_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._pending_reduce_scatters: 

        self.fsdp_layers: list[torch.nn.Module] = []
        self._replicate_parameters: list[torch.nn.Parameter] = []
        self._find_fsdp_layers_and_replicated_parameters()
        self._create_layer_states()
        self._register_forward_hooks()
        self._register_backward_hooks()

    def _broadcast_initial_state(self) -> None:
        """
        Broadcast the initial state of the model to all the ranks.
        """
        if self.world_size == 1:
            return
        
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0)
            
    def _find_fsdp_layers_and_replicated_parameters(self) -> None:
        """
        Find all the layers that are going to be sharded.
        Also find all the parameters that are going to be replicated.
        """
        for submodule in self.module.modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                self.fsdp_layers.append(submodule)
            else:
                for param in submodule.parameters(recurse=False):
                    if param.requires_grad:
                        self._replicate_parameters.append(param)
    
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

    def _get_local_shard(self, param: torch.nn.Parameter) -> tuple[torch.Tensor, ShardMetadata]:
        """
        Get the local shard of the parameter and the metadata.
        """
        metadata = self._get_shard_metadata(param)

        flattened_param = param.detach().flatten()

        if metadata.padded_num_elements > metadata.num_elements:
            padding = torch.zeros(metadata.padded_num_elements - metadata.num_elements, dtype=param.dtype, device=param.device)
            flattened_param = torch.cat([flattened_param, padding])

        local_shard = flattened_param[metadata.start:metadata.end].clone()
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

            local_shard, metadata = self._get_local_shard(param)
            param_state = FSDPParamState(
                name=param_name,
                metadata=metadata,
                local_param=torch.nn.Parameter(local_shard),
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
        for layer in self.fsdp_layers:
            self._layer_states[layer] = self._create_layer_state(layer)

    def _all_gather_param(self, local_param: torch.nn.Parameter, metadata: ShardMetadata) -> torch.Tensor:
        """
        All-gather the parameter.
        We need to all-gather the parameter for the forward and backward pass.
        We need to discard the padding elements.
        """
        local_shard = local_param.detach()

        if self.world_size == 1:
            full_flattened_param = local_shard
        else:
            gathered_shards = [torch.empty_like(local_shard) for _ in range(self.world_size)]
            dist.all_gather(gathered_shards, local_shard, async_op=False)
            full_flattened_param = torch.cat(gathered_shards, dim=0)

        full_flattened_param = full_flattened_param[:metadata.num_elements]
        return full_flattened_param.view(metadata.shape)

    def _gather_layer_params(self, layer: torch.nn.Module) -> None:
        for param_name, param_state in self._layer_states[layer].param_states.items():
            local_param = getattr(layer, param_name, None)
            metadata = param_state.metadata
            if local_param is None or metadata is None:
                continue
            param_state.local_param = local_param
            param_state.full_param = torch.nn.Parameter(self._all_gather_param(local_param, metadata))
            param_state.full_param.register_post_accumulate_grad_hook(self._make_reduce_scatter_hook(layer, param_name))
            setattr(layer, param_name, param_state.full_param)

    def _pre_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        """
        Pre-forward hook for the FSDP layers.
        We should do the all-gather of the parameters for the forward pass.
        """
        self._gather_layer_params(layer)

    def _post_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...], outputs: torch.Tensor):
        """
        Post-forward hook for the FSDP layers.
        We should shard the parameters again after the forward pass.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            setattr(layer, param_name, param_state.local_param)
            param_state.full_param.data = torch.empty(0, dtype=param_state.full_param.dtype, device=param_state.full_param.device)

    def _register_forward_hooks(self) -> None:
        """
        Register the forward hooks for the FSDP layers.
        """
        for layer in self.fsdp_layers:
            self._forward_hook_handles.append(layer.register_forward_pre_hook(self._pre_forward_hook))
            self._forward_hook_handles.append(layer.register_forward_hook(self._post_forward_hook))


    def _reduce_scatter_grad(self, full_grad: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
        """
        Reduce-scatter the gradient.
        """
        flattened_grad = full_grad.flatten()
        if metadata.padded_num_elements > metadata.num_elements:
            padding = torch.zeros(metadata.padded_num_elements - metadata.num_elements, dtype=full_grad.dtype, device=full_grad.device)
            flattened_grad = torch.cat([flattened_grad, padding])

        if self.world_size == 1:
            local_grad = flattened_grad
        else:
            local_grad = torch.empty(metadata.shard_size, dtype=full_grad.dtype, device=full_grad.device)
            dist.reduce_scatter_tensor(output=local_grad, input=flattened_grad, op=dist.ReduceOp.SUM, async_op=False)
            local_grad.div_(self.world_size)
        return local_grad

    def _copy_grad_to_local_parameter(self, full_param: torch.nn.Parameter, local_param: torch.nn.Parameter,
                                    metadata: ShardMetadata) -> None:
        """
        Reduce scatter the gradient to the local parameter.
        """
        if full_param.grad is not None:
            local_grad = self._reduce_scatter_grad(full_param.grad, metadata)
            if local_param.grad is None:
                local_param.grad = local_grad
            else:
                local_param.grad.copy_(local_grad)
        else:
            local_param.grad = None

    def _make_reduce_scatter_hook(self, layer: torch.nn.Module, param_name: str):
        """
        Return a post-accumulate-grad hook that reduce scatters the full
        weight's gradient into the local shard, then frees the full weight.
        """
        def hook(param: torch.nn.Parameter):
            self._copy_grad_to_local_parameter(param, self._layer_states[layer].param_states[param_name].local_param, self._layer_states[layer].param_states[param_name].metadata)
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            param.grad = None

        return hook

    def _regather_layer_params(self, layer: torch.nn.Module) -> None:
        """
        Rematerialize the full parameters for layer just before the backward pass.
        These are the same full_param autograd saved.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if param_state.full_param is None:
                continue
            param_state.full_param.data = self._all_gather_param(param_state.local_param, param_state.metadata)
            

    def _pre_backward_hook(self, layer: torch.nn.Module, grad_output: torch.Tensor):
        """
        Pre-backward hook for the FSDP layers.
        We should regather the parameters before the backward pass.
        """
        self._regather_layer_params(layer)

    def _register_backward_hooks(self) -> None:
        """
        Register the backward hooks for the FSDP layers.
        """
        for layer in self.fsdp_layers:
            self._backward_hook_handles.append(layer.register_full_backward_pre_hook(self._pre_backward_hook))


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
            

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self._sync_grads_of_replicated_parameters()
