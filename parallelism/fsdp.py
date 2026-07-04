from __future__ import annotations

import torch
import torch.distributed as dist

from typing import TypedDict


class ShardMetadata(TypedDict):
    num_elements: int
    shard_size: int
    padded_num_elements: int
    shape: torch.Size
    start: int
    end: int

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

        self.fsdp_layers: list[torch.nn.Module] = []
        self._find_fsdp_layers()
        self._shard_fsdp_layers()

    def _find_fsdp_layers(self):
        """
        Find all the layers that are going to be sharded.
        """
        for submodule in self.module.modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                self.fsdp_layers.append(submodule)

    def _shard_fsdp_layers(self):
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

    def _free_layer_parameters(self, layer: torch.nn.Module) -> None:
        """
        Free the parameters of the layer.
        """
        for param_name, local_param in layer._fsdp_local_params.items():
            setattr(layer, param_name, local_param)

        layer._fsdp_local_params.clear()
        

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
        for layer in self.fsdp_layers:
            self._gather_layer_params(layer)
        try:
            return self.module(*inputs, **kwargs)
        finally:
            for layer in self.fsdp_layers:
                self._free_layer_parameters(layer)

    def finish_gradient_synchronization(self):
        pass

if __name__ == "__main__":
    model = torch.nn.Sequential(
        torch.nn.Embedding(100, 16),
        torch.nn.Linear(16, 32),
        torch.nn.LayerNorm(32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )

    fsdp = FSDP(model)

    print("num fsdp layers:", len(fsdp.fsdp_layers))
    for i, layer in enumerate(fsdp.fsdp_layers):
        print(i, type(layer).__name__)
        for param_name, metadata in layer._fsdp_param_metadata.items():
            print(f"    {param_name}: {metadata}")

    print("\nInspect a single Linear layer:")
    linear_layer = fsdp.fsdp_layers[1]
    print("repr:", linear_layer)
    print("in_features:", linear_layer.in_features, "out_features:", linear_layer.out_features)
    for param_name, param in linear_layer.named_parameters():
        print(f"    {param_name}: shape={tuple(param.shape)} numel={param.numel()}")
        print(f"        values={param.data.flatten()[:8].tolist()} ...")
    print("    stored shard metadata:", linear_layer._fsdp_param_metadata)

    print("\nShard test:")

    weight = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))

    for rank in range(4):
        fsdp.rank = rank
        fsdp.world_size = 4

        shard, metadata = fsdp._get_local_shard(weight)

        print(
            f"rank={rank}",
            "shard=", shard.tolist(),
            "metadata=", metadata,
        )