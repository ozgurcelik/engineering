"""
Distributed correctness test for the FSDP container in ``fsdp.py``.

This is a test-first milestone: the current partial FSDP implementation may fail
this test until the later milestones add initialization sync, replicated-parameter
gradient sync, and the full gather/free lifecycle.

Run directly:

    conda run -n distvenv python parallelism/test_fsdp.py
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from fsdp import FSDP


WORLD_SIZE = 2
NUM_STEPS = 3
DATASET_SIZE = 12
VOCAB_SIZE = 23
SEQ_LEN = 5
EMBED_DIM = 8
HIDDEN_DIM = 16
OUT_DIM = 7


def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12357")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


class TinyFSDPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, OUT_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = torch.relu(self.proj(x))
        x = self.norm(x)
        return self.out(x)


def make_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    tokens = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(DATASET_SIZE, SEQ_LEN),
        generator=generator,
    )
    targets = torch.randn(DATASET_SIZE, SEQ_LEN, OUT_DIM, generator=generator)
    return tokens, targets


def gather_full_fsdp_named_parameters(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    """Reconstruct full parameters from FSDP local shards on every rank."""
    full_params: dict[str, torch.Tensor] = {}

    for module_name, layer in fsdp_model.module.named_modules():
        metadata_by_param = getattr(layer, "_fsdp_param_metadata", None)
        if not metadata_by_param:
            continue

        for param_name, metadata in metadata_by_param.items():
            local_param = getattr(layer, param_name)
            local_shard = local_param.detach()

            if fsdp_model.world_size == 1:
                full_flat = local_shard
            else:
                gathered_shards = [
                    torch.empty_like(local_shard)
                    for _ in range(fsdp_model.world_size)
                ]
                dist.all_gather(gathered_shards, local_shard)
                full_flat = torch.cat(gathered_shards, dim=0)

            full_param = full_flat[: metadata["num_elements"]].view(metadata["shape"])
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            full_params[full_name] = full_param.detach().clone()

    for name, param in fsdp_model.module.named_parameters():
        if name not in full_params:
            full_params[name] = param.detach().clone()

    return full_params


def assert_rank0_matches_reference(
    reference_model: nn.Module,
    fsdp_model: FSDP,
    step: int,
) -> None:
    fsdp_params = gather_full_fsdp_named_parameters(fsdp_model)

    if dist.get_rank() != 0:
        return

    reference_params = dict(reference_model.named_parameters())
    assert reference_params.keys() == fsdp_params.keys()

    for name, reference_param in reference_params.items():
        torch.testing.assert_close(
            fsdp_params[name],
            reference_param.detach(),
            rtol=1e-5,
            atol=1e-6,
            msg=f"FSDP parameter {name!r} diverged from reference at step {step}",
        )


def run_test(rank: int, world_size: int) -> None:
    setup(rank, world_size)

    try:
        torch.manual_seed(rank)
        reference_model = TinyFSDPModel()
        fsdp_model = FSDP(deepcopy(reference_model))

        optimizer_kwargs = {"lr": 0.03}
        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(),
            **optimizer_kwargs,
        )
        fsdp_optimizer = torch.optim.SGD(
            fsdp_model.parameters(),
            **optimizer_kwargs,
        )
        loss_fn = nn.MSELoss()

        all_tokens, all_targets = make_dataset()
        assert DATASET_SIZE % world_size == 0
        local_batch_size = DATASET_SIZE // world_size

        for step in range(NUM_STEPS):
            reference_optimizer.zero_grad()
            fsdp_optimizer.zero_grad()

            reference_outputs = reference_model(all_tokens)
            reference_loss = loss_fn(reference_outputs, all_targets)
            reference_loss.backward()
            reference_optimizer.step()

            offset = rank * local_batch_size
            local_tokens = all_tokens[offset : offset + local_batch_size]
            local_targets = all_targets[offset : offset + local_batch_size]

            fsdp_outputs = fsdp_model(local_tokens)
            fsdp_loss = loss_fn(fsdp_outputs, local_targets)
            fsdp_loss.backward()
            fsdp_model.finish_gradient_synchronization()
            fsdp_optimizer.step()

            assert_rank0_matches_reference(reference_model, fsdp_model, step)

        if rank == 0:
            print(f"FSDP matched the reference model for {NUM_STEPS} steps.")
    finally:
        cleanup()


def main() -> None:
    mp.spawn(run_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
