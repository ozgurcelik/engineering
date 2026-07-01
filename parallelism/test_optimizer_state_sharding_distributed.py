import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from optimizer_state_sharding import OptimizerStateSharding


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_test(rank, world_size):
    setup(rank, world_size)

    try:
        torch.manual_seed(42)

        reference_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        sharded_model = deepcopy(reference_model)

        optimizer_kwargs = {
            "lr": 0.1,
            "weight_decay": 0.1,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        reference_optimizer = torch.optim.AdamW(
            reference_model.parameters(),
            **optimizer_kwargs,
        )
        sharded_optimizer = OptimizerStateSharding(
            sharded_model.parameters(),
            torch.optim.AdamW,
            **optimizer_kwargs,
        )

        for _ in range(10):
            reference_optimizer.zero_grad()
            sharded_model.zero_grad()

            inputs = torch.rand(32, 4)
            labels = torch.rand(32, 2)

            reference_outputs = reference_model(inputs)
            sharded_outputs = sharded_model(inputs)

            reference_loss = ((labels - reference_outputs) ** 2).sum()
            sharded_loss = ((labels - sharded_outputs) ** 2).sum()

            reference_loss.backward()
            sharded_loss.backward()

            reference_optimizer.step()
            sharded_optimizer.step()

        for (reference_name, reference_param), (
            sharded_name,
            sharded_param,
        ) in zip(
            reference_model.named_parameters(),
            sharded_model.named_parameters(),
        ):
            assert reference_name == sharded_name
            torch.testing.assert_close(
                reference_param.detach(),
                sharded_param.detach(),
                rtol=1e-5,
                atol=1e-6,
            )

        print(
            f"rank={rank}: sharded optimizer matches AdamW reference; "
            f"local_params={sum(len(g['params']) for g in sharded_optimizer.param_groups)}"
        )
    finally:
        cleanup()


def main():
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
