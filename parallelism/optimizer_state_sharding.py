from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer

"""
In naive DDP, we do
1. Backward pass -> local grads
2. All-reduce grads (2.#params communications) -> everyone has the full averaged gradient
3. Every rank updates all the parameters

So, in naive DDP, every rank keeps the full optimizer state.

In zero-1, we do
1. Backward pass -> local grads
2. Reduce scatter grads (#params communications) -> each rank gets the gradient for its own shard
3. Each rank updates its own shard of the parameters
4. all-gather the updated parameters (#params communications) -> everyone has the full updated parameters

In this version of the optimizer sharding, we do
1. Backward pass -> local grads
2. all-reduce grads (2.#params communications)
3. Each rank's sharded optimizer updates its own shard of the parameters
4. broadcast the updated parameters (#params communications) -> model resynced

So this version is less communication efficient than zero-1, but still gives us 1/N optimizer state memory savings.
"""

"""
We pass the optimizer class instead of the actual optimizer instance because
the OptimizerStateSharding will determine which rank will optimize which parameters.
This is the purpose of the local_optimizer
"""

class OptimizerStateSharding(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        """
        opt = OptimizerStateSharding(
            model.parameters(),     # -> params
            torch.optim.AdamW,      # -> optimizer_cls
            lr=1e-3,                # ┐
            weight_decay=0.01,      # ├─ all collected into **kwargs
            betas=(0.9, 0.95),      # ┘
        )
        """
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self.local_optimizer: Optimizer | None = None

        self.full_params: list[torch.nn.Parameter] = []
        # global ids of the params
        self.pid_to_gid: dict[int, int] = {}

        super().__init__(params, defaults=self.optimizer_kwargs)

    
    def assign_global_ids(self, params_list: list[torch.nn.Parameter]) -> None:
        for param in params_list:
            pid = id(param)
            if pid not in self.pid_to_gid:
                self.pid_to_gid[pid] = len(self.pid_to_gid)
                self.full_params.append(param)

    def get_rank_for_a_parameter(self, param: torch.nn.Parameter) -> int:
        pid = id(param)
        return self.pid_to_gid[pid] % self.world_size

    def get_local_params_from_a_group(self, group_of_params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        local_params = []
        self.assign_global_ids(group_of_params)
        for param in group_of_params:
            param_rank = self.get_rank_for_a_parameter(param)
            if param_rank == self.rank:
                local_params.append(param)
        return local_params

    def add_param_group(self, param_group):

        # Get the params from the param group
        if "params" not in param_group:
            raise ValueError("param_group must contain a 'params' key")
        params = param_group["params"]

        full_params_list = list(params)

        local_params = self.get_local_params_from_a_group(full_params_list)

        # Now we need to copy the param group and replace the params with the local params
        local_param_group = param_group.copy()
        local_param_group["params"] = local_params

        if len(local_params) == 0:
            return

        # Add the local param group to the optimizer wrapper
        # So the optimizer will keep track of the local param group
        super().add_param_group(local_param_group)

        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls(self.param_groups, **self.optimizer_kwargs)
        else:
            self.local_optimizer.add_param_group(local_param_group)
        

    def step(self, closure=None, **kwargs):
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure=closure, **kwargs)
        
        if (
            dist.is_available()
            and dist.is_initialized()
            and self.world_size > 1
        ):
            with torch.no_grad():
                for param in self.full_params:
                    param_rank = self.get_rank_for_a_parameter(param)
                    dist.broadcast(param, src=param_rank)

        return loss

if __name__ == "__main__":
    from unittest.mock import patch
    import torch.nn as nn

    def test_broadcast_calls(rank):
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 2),
        )

        broadcast_sources = []

        def mock_broadcast(tensor, src):
            broadcast_sources.append(src)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_rank", return_value=rank),
            patch.object(dist, "broadcast", side_effect=mock_broadcast),
        ):
            opt = OptimizerStateSharding(
                model.parameters(),
                torch.optim.SGD,
                lr=0.1,
            )

            before = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }

            loss = model(torch.ones(3, 4)).sum()

            model.zero_grad()
            loss.backward()
            opt.step()

        print(f"\nrank={rank}")
        print("broadcast sources:", broadcast_sources)

        assert broadcast_sources == [0, 1, 0, 1]

        for name, param in model.named_parameters():
            owner = opt.get_rank_for_a_parameter(param)
            changed = not torch.equal(before[name], param.detach())

            print(
                f"{name:8} owner={owner}, "
                f"locally changed={changed}"
            )

            assert changed == (owner == rank)

    test_broadcast_calls(rank=0)
    test_broadcast_calls(rank=1)

    print("\nMock broadcast tests passed.")
