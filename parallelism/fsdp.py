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

        self.fsdp_modules: list[torch.nn.Module] = []
        self._find_fsdp_modules()
        self._shard_fsdp_modules()

    def _find_fsdp_modules(self):
        """
        Find all the modules that are going to be sharded.
        """
        for submodule in self.module.modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                self.fsdp_modules.append(submodule)

    def _shard_fsdp_modules(self):
        """
        Shard the parameters of the FSDP modules.
        """
        for module in self.fsdp_modules:
            module._fsdp_param_metadata: dict[str, ShardMetadata] = {}
            module._fsdp_local_params: dict[str, torch.nn.Parameter] = {}
            self._shard_module_parameters(module)
    
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

    def _shard_module_parameters(self, module: torch.nn.Module) -> None:
        """
        Shard the parameters of the module.
        Linear has weight and bias, Embedding has weight.
        """
        for param_name in ("weight", "bias"):
            param = getattr(module, param_name, None)
            if param is None:
                continue

            local_shard, metadata = self._get_local_shard(param)
            module._fsdp_param_metadata[param_name] = metadata

            setattr(module, param_name, torch.nn.Parameter(local_shard))

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

    def _save_local_parameters(self, module: torch.nn.Module, param_name: str, param: torch.nn.Parameter) -> None:
        """
        Save the local shard of the parameter so that we don't lose it during the forward pass.
        """
        local_shard, metadata = self._get_local_shard(param)
        module._fsdp_local_params[param_name] = torch.nn.Parameter(local_shard)

    def _free_module_parameters(self, module: torch.nn.Module) -> None:
        """
        Free the parameters of the module.
        """
        return None
        

    def _gather_module_params(self, module: torch.nn.Module) -> None:
        for param_name in ("weight", "bias"):
            local_param = getattr(module, param_name, None)
            metadata = module._fsdp_param_metadata.get(param_name)
            if local_param is None or metadata is None:
                continue
            full_param = self._all_gather_param(local_param, metadata)
            setattr(module, param_name, torch.nn.Parameter(full_param))
            

    def forward(self, *inputs, **kwargs):
        for module in self.fsdp_modules:
            self._gather_module_params(module)
        return self.module(*inputs, **kwargs)

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

    print("num fsdp modules:", len(fsdp.fsdp_modules))
    for i, module in enumerate(fsdp.fsdp_modules):
        print(i, type(module).__name__)

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