"""
Correctness test for the DDP containers in ``ddp.py``.

  * A single-process "non-parallel" model sees the FULL batch every step.
  * A DDP model is replicated across ``world_size`` ranks, and each rank sees a
    DISJOINT shard of that same batch. Gradients are all-reduced (averaged)
    across ranks, so the effective gradient equals the full-batch gradient.

Therefore, after every optimizer step, the DDP model on rank 0 must match the
non-parallel baseline bit-for-bit (the broadcast at construction time makes the
DDP replicas start from rank 0's weights).

Run directly (spawns processes itself):

    python test_ddp.py
"""

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from ddp import DDP, DDP_Hook, DDP_Bucket


WORLD_SIZE = 2
NUM_STEPS = 5
DATASET_SIZE = 20
IN_FEATURES = 10
OUT_FEATURES = 5

DDP_VARIANTS = {
    "DDP": DDP,
    "DDP_Hook": DDP_Hook,
    "DDP_Bucket": DDP_Bucket,
}


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    # gloo is the simplest backend for CPU-only correctness tests.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self, in_features: int = IN_FEATURES, out_features: int = OUT_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_features),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset():
    """Deterministic dataset, identical on every rank (same seed)."""
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(DATASET_SIZE, IN_FEATURES, generator=g)
    y = torch.randn(DATASET_SIZE, OUT_FEATURES, generator=g)
    return x, y


def _run(rank: int, world_size: int, variant_name: str):
    setup(rank, world_size)
    try:
        ddp_cls = DDP_VARIANTS[variant_name]

        # Seed per-rank so each rank's *initial* model differs. The DDP container
        # then broadcasts rank 0's weights to everyone, so they all converge to
        # rank 0's starting point. This also proves the broadcast actually runs.
        torch.manual_seed(rank)
        non_parallel_model = ToyModel()

        ddp_model = ddp_cls(deepcopy(non_parallel_model))

        # After construction, every rank's DDP weights must equal rank 0's.
        # On rank 0 that means DDP == its own non_parallel baseline.
        if rank == 0:
            for p_np, p_ddp in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                assert torch.allclose(p_np, p_ddp), "rank 0 weights changed by broadcast"

        loss_fn = nn.MSELoss()
        opt_np = torch.optim.SGD(non_parallel_model.parameters(), lr=0.1)
        opt_ddp = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        all_x, all_y = make_dataset()
        assert DATASET_SIZE % world_size == 0
        local_bs = DATASET_SIZE // world_size

        for step in range(NUM_STEPS):
            opt_np.zero_grad()
            opt_ddp.zero_grad()

            # Non-parallel baseline: full batch.
            np_out = non_parallel_model(all_x)
            np_loss = loss_fn(np_out, all_y)
            np_loss.backward()
            opt_np.step()

            # DDP: each rank takes a disjoint shard of the SAME full batch.
            offset = rank * local_bs
            local_x = all_x[offset : offset + local_bs]
            local_y = all_y[offset : offset + local_bs]

            ddp_out = ddp_model(local_x)
            ddp_loss = loss_fn(ddp_out, local_y)
            ddp_loss.backward()
            ddp_model.finish_gradient_synchronization()
            opt_ddp.step()

            # Rank 0's DDP model must match the full-batch baseline every step.
            if rank == 0:
                for p_np, p_ddp in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                    assert torch.allclose(p_np, p_ddp, atol=1e-5), (
                        f"[{variant_name}] mismatch at step {step}"
                    )

            # Reshuffle with a shared seed so shards stay disjoint AND the union
            # of all shards still equals the baseline's full batch.
            torch.manual_seed(1000 + step)
            perm = torch.randperm(DATASET_SIZE)
            all_x = all_x[perm]
            all_y = all_y[perm]

        if rank == 0:
            print(f"[{variant_name}] OK: DDP matched non-parallel baseline for {NUM_STEPS} steps.")
    finally:
        cleanup()


def run_variant(variant_name: str, world_size: int = WORLD_SIZE):
    mp.spawn(_run, args=(world_size, variant_name), nprocs=world_size, join=True)


if __name__ == "__main__":
    for name in DDP_VARIANTS:
        run_variant(name)
