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
        layer_state = fsdp_model._layer_states.get(layer)
        if layer_state is None:
            continue

        for param_name, param_state in layer_state.param_states.items():
            local_shard = param_state.local_param.detach()
            metadata = param_state.metadata

            if fsdp_model.world_size == 1:
                full_flat = local_shard
            else:
                gathered_shards = [
                    torch.empty_like(local_shard)
                    for _ in range(fsdp_model.world_size)
                ]
                dist.all_gather(gathered_shards, local_shard)
                full_flat = torch.cat(gathered_shards, dim=0)

            full_param = full_flat[: metadata.num_elements].view(metadata.shape)
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


def assert_layers_are_sharded(
    fsdp_model: FSDP,
    *,
    expect_full_params: bool,
) -> None:
    for layer, layer_state in fsdp_model._layer_states.items():
        for param_name, param_state in layer_state.param_states.items():
            actual_param = getattr(layer, param_name)
            assert actual_param is param_state.local_param
            assert tuple(actual_param.shape) == tuple(param_state.local_param.shape)
            if expect_full_params:
                assert param_state.full_param is not None
            else:
                assert param_state.full_param is None


def _full_param_is_freed(param_state) -> bool:
    """A gathered full weight counts as freed when its underlying storage is
    empty. We keep the Parameter object alive (so backward can refill it in a
    later milestone), but its data must no longer occupy memory."""
    full = param_state.full_param
    if full is None:
        return True
    if full.numel() == 0:
        return True
    return full.untyped_storage().size() == 0


def assert_full_params_freed_after_forward(fsdp_model: FSDP) -> None:
    for layer, layer_state in fsdp_model._layer_states.items():
        for param_name, param_state in layer_state.param_states.items():
            actual_param = getattr(layer, param_name)
            assert actual_param is param_state.local_param, (
                f"layer attribute {param_name!r} should point back at the local shard"
            )
            assert tuple(actual_param.shape) == tuple(param_state.local_param.shape)
            assert _full_param_is_freed(param_state), (
                f"full weight for {param_name!r} was not freed after forward"
            )


def broadcast_reference_model(reference_model: nn.Module) -> None:
    with torch.no_grad():
        for param in reference_model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in reference_model.buffers():
            dist.broadcast(buffer.data, src=0)


def run_forward_lifecycle_test(rank: int, world_size: int) -> None:
    """Milestone 1 verification.

    Checks three things about the forward pass:
      1. The FSDP forward output still matches the reference model (gather works).
      2. Every gathered full weight is freed once its layer's forward is done.
      3. At most one FSDP layer's full weight is materialized at any instant
         (no prefetch yet), proving we gather-then-free layer by layer.
    """
    setup(rank, world_size)

    try:
        torch.manual_seed(rank)
        reference_model = TinyFSDPModel()
        broadcast_reference_model(reference_model)
        fsdp_model = FSDP(deepcopy(reference_model))

        # Measure how many FSDP layers hold a *materialized* (non-empty) full
        # weight at the same time, by inspecting real storage state instead of
        # intercepting method calls (so it doesn't matter how the free path is
        # factored inside fsdp.py). We sample right after each layer's weights
        # have been gathered: our pre-hook is registered *after* FSDP's own
        # pre-hook, so it runs second and sees the freshly gathered state.
        # Without prefetching, the peak must be exactly 1: by the time layer i
        # is gathered, layer i-1 has already been freed in its post-hook.
        peak_live = {"value": 0}

        def count_materialized_layers() -> int:
            materialized = 0
            for layer_state in fsdp_model._layer_states.values():
                if any(
                    not _full_param_is_freed(param_state)
                    for param_state in layer_state.param_states.values()
                ):
                    materialized += 1
            return materialized

        def sample_after_gather(_layer, _inputs):
            peak_live["value"] = max(peak_live["value"], count_materialized_layers())

        for fsdp_layer in fsdp_model.fsdp_layers:
            fsdp_layer.register_forward_pre_hook(sample_after_gather)

        tokens, _ = make_dataset()
        local_batch_size = DATASET_SIZE // world_size
        offset = rank * local_batch_size
        local_tokens = tokens[offset : offset + local_batch_size]

        reference_outputs = reference_model(local_tokens)
        fsdp_outputs = fsdp_model(local_tokens)

        torch.testing.assert_close(
            fsdp_outputs,
            reference_outputs,
            rtol=1e-5,
            atol=1e-6,
        )
        assert_full_params_freed_after_forward(fsdp_model)

        assert peak_live["value"] == 1, (
            f"expected at most one materialized full weight at a time, "
            f"but peak was {peak_live['value']}"
        )

        if rank == 0:
            print(
                "Milestone 1 OK: forward matched reference, full weights freed, "
                f"peak simultaneous full weights = {peak_live['value']}."
            )
    finally:
        cleanup()


def install_backward_peak_probe(fsdp_model: FSDP, peak_live: dict) -> None:
    """Sample how many FSDP layers are simultaneously materialized during the
    backward pass. We wrap `_regather_layer_params`, which runs right after a
    layer's weight has been re-gathered for its backward. By that point every
    layer processed earlier in the backward must already be freed, so the peak
    stays at 1. If a layer forgets to free, upstream layers pile up and the
    peak climbs.

    This is best-effort instrumentation: if the method was renamed, we skip it
    and rely on the end-to-end correctness check instead.
    """
    method_name = (
        "_gather_layer_params_for_backward"
        if hasattr(fsdp_model, "_gather_layer_params_for_backward")
        else "_regather_layer_params"
    )
    original = getattr(fsdp_model, method_name, None)
    if original is None:
        return

    def count_materialized_layers() -> int:
        materialized = 0
        for layer_state in fsdp_model._layer_states.values():
            if any(
                not _full_param_is_freed(param_state)
                for param_state in layer_state.param_states.values()
            ):
                materialized += 1
        return materialized

    def probed(layer):
        result = original(layer)
        peak_live["value"] = max(peak_live["value"], count_materialized_layers())
        return result

    setattr(fsdp_model, method_name, probed)


def _count_materialized_layers(fsdp_model: FSDP) -> int:
    materialized = 0
    for layer_state in fsdp_model._layer_states.values():
        if any(
            not _full_param_is_freed(param_state)
            for param_state in layer_state.param_states.values()
        ):
            materialized += 1
    return materialized


def install_forward_peak_probe(fsdp_model: FSDP, fwd_peak: dict) -> None:
    """Sample how many FSDP layers hold a materialized full weight during the
    forward pass. With the 'start gathering layer i once layer i-2 has finished
    its forward' prefetch rule, exactly one layer is prefetched ahead of the
    one currently in use, so the peak is 2 (for a model with >= 3 FSDP layers).

    - No prefetch (just-in-time gather) would give a peak of 1.
    - Prefetching everything up front would give a peak of #layers (here 3).

    We register our own pre/post forward hooks; they run after FSDP's own hooks
    (registration order), so they observe the post-use / post-free+prefetch
    state. A prefetched-but-not-yet-used layer counts as materialized because
    its full weight storage is already allocated for the in-flight all-gather.
    """

    def sample(*_args):
        fwd_peak["value"] = max(fwd_peak["value"], _count_materialized_layers(fsdp_model))

    for layer in fsdp_model.fsdp_layers:
        layer.register_forward_pre_hook(sample)
        layer.register_forward_hook(sample)


def install_async_all_gather_probe(ag_calls: dict) -> callable:
    """Record whether the weight all-gather is issued asynchronously (so it can
    be prefetched and overlap with compute). Detects both the flat
    `all_gather_into_tensor` and the list-based `all_gather` APIs. Returns a
    restore() callable."""
    original_into = getattr(dist, "all_gather_into_tensor", None)
    original_list = getattr(dist, "all_gather", None)

    def make_traced(original):
        def traced(*args, **kwargs):
            if kwargs.get("async_op", False):
                ag_calls["async"] += 1
            else:
                ag_calls["sync"] += 1
            return original(*args, **kwargs)

        return traced

    if original_into is not None:
        dist.all_gather_into_tensor = make_traced(original_into)
    if original_list is not None:
        dist.all_gather = make_traced(original_list)

    def restore() -> None:
        if original_into is not None:
            dist.all_gather_into_tensor = original_into
        if original_list is not None:
            dist.all_gather = original_list

    return restore


def install_async_reduce_scatter_probe(async_calls: dict) -> callable:
    """Record whether reduce-scatter is issued asynchronously. Returns a
    restore() callable that undoes the patch. Milestone 3 requires the
    gradient reduce-scatter to run with async_op=True so it can overlap with
    the remaining backward compute."""
    original = dist.reduce_scatter_tensor

    def traced(*args, **kwargs):
        if kwargs.get("async_op", False):
            async_calls["async"] += 1
        else:
            async_calls["sync"] += 1
        return original(*args, **kwargs)

    dist.reduce_scatter_tensor = traced

    def restore() -> None:
        dist.reduce_scatter_tensor = original

    return restore


def run_training_test(rank: int, world_size: int) -> None:
    setup(rank, world_size)

    try:
        torch.manual_seed(rank)
        reference_model = TinyFSDPModel()
        broadcast_reference_model(reference_model)
        fsdp_model = FSDP(deepcopy(reference_model))

        peak_live = {"value": 0}
        install_backward_peak_probe(fsdp_model, peak_live)

        fwd_peak = {"value": 0}
        install_forward_peak_probe(fsdp_model, fwd_peak)

        async_calls = {"async": 0, "sync": 0}
        restore_reduce_scatter = install_async_reduce_scatter_probe(async_calls)

        all_gather_calls = {"async": 0, "sync": 0}
        restore_all_gather = install_async_all_gather_probe(all_gather_calls)

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

            # After sync, weights must be back to sharded local shards and every
            # gathered full weight must be freed (backward re-gather cleaned up).
            assert_full_params_freed_after_forward(fsdp_model)
            fsdp_optimizer.step()

            assert_rank0_matches_reference(reference_model, fsdp_model, step)

        restore_reduce_scatter()
        restore_all_gather()

        assert peak_live["value"] <= 1, (
            f"expected at most one materialized full weight during backward, "
            f"but peak was {peak_live['value']} (a layer likely did not free "
            f"its re-gathered weight)"
        )

        assert async_calls["async"] > 0, (
            "expected the gradient reduce-scatter to be issued with "
            "async_op=True so it can overlap with backward compute, but every "
            f"reduce-scatter was synchronous (async={async_calls['async']}, "
            f"sync={async_calls['sync']})"
        )

        assert all_gather_calls["async"] > 0, (
            "expected the weight all-gather to be issued with async_op=True so "
            "it can be prefetched and overlap with compute, but every "
            f"all-gather was synchronous (async={all_gather_calls['async']}, "
            f"sync={all_gather_calls['sync']})"
        )

        assert fwd_peak["value"] == 2, (
            f"expected forward peak of exactly 2 materialized layers with the "
            f"'gather layer i after layer i-2 finished forward' prefetch rule "
            f"(one in use + one prefetched ahead), but peak was "
            f"{fwd_peak['value']}. A peak of 1 means no prefetch; a peak of 3 "
            f"means everything was gathered up front."
        )

        if rank == 0:
            print(
                f"Milestone 4 OK: FSDP matched the reference for {NUM_STEPS} steps; "
                f"forward peak materialized layers = {fwd_peak['value']}; "
                f"async all-gathers = {all_gather_calls['async']}; "
                f"backward peak = {peak_live['value']}; "
                f"async reduce-scatters = {async_calls['async']}."
            )
    finally:
        cleanup()


def main() -> None:
    # Milestone 4: forward weights are prefetched asynchronously using the
    # 'start gathering layer i once layer i-2 finished its forward' rule, so at
    # most two layers are materialized at once during the forward pass.
    mp.spawn(
        run_training_test,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
