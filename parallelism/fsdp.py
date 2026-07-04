from __future__ import annotations

import torch
import torch.distributed as dist

from typing import TypedDict

from dataclasses import dataclass, field


class ShardMetadata(TypedDict):
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

        self.fsdp_layers: list[torch.nn.Module] = []
        self._replicate_parameters: list[torch.nn.Parameter] = []
        self._find_fsdp_layers_and_replicated_parameters()
        self._shard_fsdp_layers()

        self._has_full_params = False

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

    def _create_layer_state(self, layer: torch.nn.Module) -> FSDPLayerState:
        """
        Create the state for the layer.
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
            
        return FSDPLayerState(
            layer=layer,
            param_states=param_states,
        )
            
            
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

    def _shard_fsdp_layers(self) -> None:
        """
        Shard the parameters of the FSDP layers.
        """
        for layer in self.fsdp_layers:
            layer._fsdp_param_metadata: dict[str, ShardMetadata] = {}
            layer._fsdp_local_params: dict[str, torch.nn.Parameter] = {}
            self._shard_layer_parameters(layer)
    
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

        if metadata["padded_num_elements"] > metadata["num_elements"]:
            padding = torch.zeros(metadata["padded_num_elements"] - metadata["num_elements"], dtype=param.dtype, device=param.device)
            flattened_param = torch.cat([flattened_param, padding])

        local_shard = flattened_param[metadata["start"]:metadata["end"]].clone()
        return local_shard, metadata

    def _shard_layer_parameters(self, layer: torch.nn.Module) -> None:
        """
        Shard the parameters of the layer.
        Linear has weight and bias, Embedding has weight.
        """
        for param_name in ("weight", "bias"):
            param = getattr(layer, param_name, None)
            if param is None:
                continue

            local_shard, metadata = self._get_local_shard(param)
            layer._fsdp_param_metadata[param_name] = metadata

            setattr(layer, param_name, torch.nn.Parameter(local_shard))

    def _reduce_scatter_grad(self, full_grad: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
        """
        Reduce-scatter the gradient.
        """
        flattened_grad = full_grad.flatten()
        if metadata["padded_num_elements"] > metadata["num_elements"]:
            padding = torch.zeros(metadata["padded_num_elements"] - metadata["num_elements"], dtype=full_grad.dtype, device=full_grad.device)
            flattened_grad = torch.cat([flattened_grad, padding])

        if self.world_size == 1:
            local_grad = flattened_grad
        else:
            local_grad = torch.empty(metadata["shard_size"], dtype=full_grad.dtype, device=full_grad.device)
            dist.reduce_scatter_tensor(output=local_grad, input=flattened_grad, op=dist.ReduceOp.SUM, async_op=False)
            local_grad.div_(self.world_size)
        return local_grad

    def _copy_grad_to_local_parameter(self, full_param: torch.nn.Parameter, local_param: torch.nn.Parameter,
                                    metadata: ShardMetadata) -> None:
        """
        Copy the gradient to the local parameter.
        """
        if full_param.grad is not None:
            local_grad = self._reduce_scatter_grad(full_param.grad, metadata)
            if local_param.grad is None:
                local_param.grad = local_grad
            else:
                local_param.grad.copy_(local_grad)
        else:
            local_param.grad = None


    def _all_gather_param(self, local_param: torch.nn.Parameter, metadata: ShardMetadata) -> torch.Tensor:
        """
        All-gather the parameter.
        We need to all-gather the parameter for the forward and backward pass.
        """
        local_shard = local_param.detach()

        if self.world_size == 1:
            full_flattened_param = local_shard
        else:
            gathered_shards = [torch.empty_like(local_shard) for _ in range(self.world_size)]
            dist.all_gather(gathered_shards, local_shard, async_op=False)
            full_flattened_param = torch.cat(gathered_shards, dim=0)

        full_flattened_param = full_flattened_param[:metadata["num_elements"]]
        return full_flattened_param.view(metadata["shape"])

    def _save_local_parameter(self, layer: torch.nn.Module, param_name: str, param: torch.nn.Parameter) -> None:
        """
        Save the local shard of the parameter so that we don't lose it during the forward pass.
        """
        layer._fsdp_local_params[param_name] = param

    def _restore_local_parameters(self, layer: torch.nn.Module) -> None:
        """
        Restore the local parameters of the layer.
        """
        for param_name, local_param in layer._fsdp_local_params.items():
            full_param = getattr(layer, param_name, None)
            if full_param is None or local_param is None:
                continue
            metadata = layer._fsdp_param_metadata.get(param_name)
            if metadata is None:
                continue
            self._copy_grad_to_local_parameter(full_param, local_param, metadata)
            setattr(layer, param_name, local_param)

        layer._fsdp_local_params.clear()

    def _free_all_layer_parameters(self) -> None:
        """
        Free the parameters of all the layers.
        """
        for layer in self.fsdp_layers:
            self._restore_local_parameters(layer)

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
        

    def _gather_layer_params(self, layer: torch.nn.Module) -> None:
        for param_name in ("weight", "bias"):
            local_param = getattr(layer, param_name, None)
            metadata = layer._fsdp_param_metadata.get(param_name)
            if local_param is None or metadata is None:
                continue
            self._save_local_parameter(layer, param_name, local_param)
            full_param = self._all_gather_param(local_param, metadata)
            setattr(layer, param_name, torch.nn.Parameter(full_param))
            

    def forward(self, *inputs, **kwargs):
        if not self._has_full_params:
            for layer in self.fsdp_layers:
                self._gather_layer_params(layer)
            self._has_full_params = True
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self._free_all_layer_parameters()
        self._sync_grads_of_replicated_parameters()
        self._has_full_params = False
