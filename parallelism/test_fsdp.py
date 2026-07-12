"""Tests for the FullyShardedDataParallel (FSDP) implementation in ``fsdp.py``.

Modeled after the Stanford CS336 assignment-2 FSDP tests
(https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_fsdp.py),
adapted to run against this repo's ``FSDP`` class as a standalone script
(no pytest / cs336_basics dependency):

    conda run -n distvenv python parallelism/test_fsdp.py

Two checks, each run for compute_dtype in {None (fp32), torch.float16}:
  * test_fsdp_correctness  - sharded FSDP matches a single-process baseline
    (with matching mixed-precision behavior when compute_dtype is set).
  * test_fsdp_gradient_sync - after sync, every parameter has a correctly
    shaped/typed gradient, and replicated (non-sharded) grads agree on all ranks.
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from fsdp import FSDP


def _cast_floating(obj, dtype: torch.dtype):
    """Cast every floating-point tensor in obj (Tensor / tuple / list) to
    dtype via a differentiable ``.to()``; leave integer/bool tensors and
    non-tensors untouched. Mirrors the helper the FSDP activation casting uses."""
    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, tuple):
        return tuple(_cast_floating(o, dtype) for o in obj)
    if isinstance(obj, list):
        return [_cast_floating(o, dtype) for o in obj]
    return obj


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm that preserves the activation dtype (upcasts to fp32 internally
    for the reduction, casts the result back). This is what lets a fp16 model
    flow low-precision activations through the norms without hitting a
    fp32/fp16 boundary at the following Linear."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(in_dtype)


class ToyFSDPModel(nn.Module):
    """Embedding + norms + linears: exercises both FSDP-sharded layers
    (Embedding/Linear) and replicated layers (RMSNorm)."""

    def __init__(self, vocab_size: int = 100, d_model: int = 64, d_ff: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm1 = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.norm2 = RMSNorm(d_ff)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.norm1(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.lm_head(x)
        return x


class ToyMLPModel(nn.Module):
    """An MLP fed *raw fp32 activations*: the very first FSDP layer is a Linear
    that receives fp32 input, so nothing casts the activation down to
    compute_dtype before it meets a low-precision weight. This is the case that
    crashes a weight-only-casting FSDP ("float != Half"), and the one that
    activation-boundary casting is meant to fix. Also contains a real fp32
    ``nn.LayerNorm`` to exercise a replicated norm on the low-precision path."""

    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 16) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden, bias=False)
        self.norm = nn.LayerNorm(d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = self.norm(x)
        return self.l2(x)


class FrozenShardedParamModel(nn.Module):
    """Contains frozen Embedding/Linear parameters, which FSDP still needs to
    all-gather for compute but must not turn into trainable local shards."""

    frozen_names = {"embedding.weight", "linear2.weight", "linear2.bias"}

    def __init__(self, vocab_size: int = 29, d_model: int = 9, d_hidden: int = 17, d_out: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.linear3 = nn.Linear(d_hidden, d_out)

        self.embedding.weight.requires_grad = False
        self.linear2.weight.requires_grad = False
        self.linear2.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).mean(dim=1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


# ---------------------------------------------------------------------------
# Adapters onto this repo's FSDP API (mirrors the CS336 adapter functions)
# ---------------------------------------------------------------------------


def get_fsdp(module: nn.Module, compute_dtype: torch.dtype | None = None) -> FSDP:
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: FSDP, optimizer: torch.optim.Optimizer) -> None:
    # optimizer is accepted to match the CS336 signature; unused here.
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    """Reconstruct full (unsharded) parameters from the local fp32 master
    shards on every rank, keyed like ``module.named_parameters()``."""
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
                gathered = [torch.empty_like(local_shard) for _ in range(fsdp_model.world_size)]
                dist.all_gather(gathered, local_shard)
                full_flat = torch.cat(gathered, dim=0)
            full = full_flat[: metadata.num_elements].view(metadata.shape)
            name = f"{module_name}.{param_name}" if module_name else param_name
            full_params[name] = full.detach().clone()

    for name, param in fsdp_model.module.named_parameters():
        if name not in full_params:
            full_params[name] = param.detach().clone()

    return full_params


# ---------------------------------------------------------------------------
# Process group helpers
# ---------------------------------------------------------------------------


def _setup_process_group(rank: int, world_size: int, backend: str = "gloo") -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "12357")
    # Set FSDP_TEST_DEVICE=cuda to run the same tests on GPU (one rank per GPU
    # over NCCL); anything else keeps the default single-host CPU/gloo path.
    use_cuda = (
        os.environ.get("FSDP_TEST_DEVICE", "cpu") == "cuda" and torch.cuda.is_available()
    )
    if use_cuda:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
        return device
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return torch.device("cpu")


def _cleanup_process_group() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Mixed-precision baseline: replicate FSDP's recipe on a single-process model
# ---------------------------------------------------------------------------


def _apply_mixed_precision_hooks(model: nn.Module, compute_dtype: torch.dtype) -> None:
    """Cast Linear/Embedding weights to compute_dtype for forward/backward,
    keep master weights + optimizer updates in fp32 (grads cast back to fp32).
    This is the single-process analogue of what sharded FSDP does."""
    for mod in model.modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue

        def make_fwd_pre(dt):
            def hook(m, inp):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dt)

            return hook

        def make_fwd_post():
            def hook(m, inp, out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None

            return hook

        mod.register_forward_pre_hook(make_fwd_pre(compute_dtype))
        mod.register_forward_hook(make_fwd_post())

        # Linear backward needs the weight in compute_dtype for grad_input.
        if isinstance(mod, nn.Linear):

            def make_bwd_pre(dt):
                def hook(m, grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dt)
                    m.weight.grad = None

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        mod.weight.register_post_accumulate_grad_hook(
            make_grad_hook(mod, isinstance(mod, nn.Linear))
        )


def _apply_boundary_mixed_precision_hooks(model: nn.Module, compute_dtype: torch.dtype) -> None:
    """Single-process reference for FSDP's *transparent activation-boundary*
    mixed precision. In addition to casting each Linear/Embedding weight to
    compute_dtype (like ``_apply_mixed_precision_hooks``), this also casts the
    layer's floating-point *inputs* to compute_dtype on entry and casts the
    floating-point *outputs* back to the incoming dtype on exit (for Embedding,
    whose input is integer, outputs are restored to the fp32 master dtype).

    This is exactly what the activation-aware FSDP does at each wrapped layer,
    so an FSDP model must match a single-process model with these hooks."""
    for mod in model.modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        master_dtype = mod.weight.dtype  # fp32

        def make_fwd_pre(dt, master):
            def hook(m, args):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dt)
                restore = master
                for a in args:
                    if torch.is_tensor(a) and a.is_floating_point():
                        restore = a.dtype
                        break
                m._restore_dtype = restore
                return _cast_floating(args, dt)

            return hook

        def make_fwd_post():
            def hook(m, args, out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None
                return _cast_floating(out, m._restore_dtype)

            return hook

        mod.register_forward_pre_hook(make_fwd_pre(compute_dtype, master_dtype))
        mod.register_forward_hook(make_fwd_post())

        if isinstance(mod, nn.Linear):

            def make_bwd_pre(dt):
                def hook(m, grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dt)
                    m.weight.grad = None

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        mod.weight.register_post_accumulate_grad_hook(
            make_grad_hook(mod, isinstance(mod, nn.Linear))
        )


# ---------------------------------------------------------------------------
# Correctness: sharded FSDP == single-process baseline
# ---------------------------------------------------------------------------


def _test_fsdp_correctness(rank: int, world_size: int, compute_dtype) -> None:
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(42)
    base_model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128).to(device)

    non_parallel_model = deepcopy(base_model)
    if compute_dtype is not None:
        _apply_mixed_precision_hooks(non_parallel_model, compute_dtype)

    fsdp_model = get_fsdp(deepcopy(base_model), compute_dtype=compute_dtype)

    loss_fn = nn.CrossEntropyLoss()
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
    non_parallel_optimizer = torch.optim.SGD(non_parallel_model.parameters(), lr=0.01)

    torch.manual_seed(123)
    batch_size = 20
    seq_len = 8
    all_input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    all_labels = torch.randint(0, 100, (batch_size,), device=device)

    local_bs = batch_size // world_size

    for step in range(3):
        fsdp_optimizer.zero_grad(set_to_none=True)
        non_parallel_optimizer.zero_grad(set_to_none=True)

        non_parallel_out = non_parallel_model(all_input_ids)
        non_parallel_loss = loss_fn(non_parallel_out[:, -1, :].float(), all_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        offset = rank * local_bs
        local_input = all_input_ids[offset : offset + local_bs]
        local_labels = all_labels[offset : offset + local_bs]
        fsdp_out = fsdp_model(local_input)
        fsdp_loss = loss_fn(fsdp_out[:, -1, :].float(), local_labels)
        fsdp_loss.backward()

        fsdp_on_after_backward(fsdp_model, fsdp_optimizer)
        fsdp_optimizer.step()

        full_params = fsdp_gather_full_params(fsdp_model)
        for name, np_param in non_parallel_model.named_parameters():
            fsdp_full = full_params[name]
            atol = 1e-6 if compute_dtype is None else 1e-4
            rtol = 1e-4
            assert torch.allclose(np_param.data, fsdp_full, atol=atol, rtol=rtol), (
                f"Step {step}: Parameter {name} mismatch. Max diff: "
                f"{(np_param.data - fsdp_full).abs().max().item()}"
            )

        torch.manual_seed(42 + step)
        perm = torch.randperm(batch_size)
        all_input_ids = all_input_ids[perm]
        all_labels = all_labels[perm]

    if rank == 0:
        tag = "fp32" if compute_dtype is None else str(compute_dtype)
        print(f"test_fsdp_correctness[{tag}]: matched the baseline for 3 steps.")

    _cleanup_process_group()


# ---------------------------------------------------------------------------
# Gradient sync: shapes, dtypes, and replicated-grad agreement across ranks
# ---------------------------------------------------------------------------


def _test_fsdp_gradient_sync(rank: int, world_size: int, compute_dtype) -> None:
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(42)
    model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128).to(device)
    fsdp_model = get_fsdp(model, compute_dtype=compute_dtype)

    # Each rank gets different data, so replicated grads must be synced.
    torch.manual_seed(rank)
    input_ids = torch.randint(0, 100, (4, 8), device=device)

    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)

    out = fsdp_model(input_ids)
    loss = out.sum()
    loss.backward()

    fsdp_on_after_backward(fsdp_model, fsdp_optimizer)

    # Every parameter must carry a gradient matching its (sharded) data shape,
    # in the fp32 master dtype, regardless of compute_dtype.
    for name, param in fsdp_model.module.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"Gradient is None for {name}"
        assert param.grad.shape == param.data.shape, (
            f"Gradient shape {param.grad.shape} != data shape {param.data.shape} for {name}"
        )
        assert param.grad.dtype == param.data.dtype, (
            f"Gradient dtype {param.grad.dtype} != data dtype {param.data.dtype} for {name}"
        )

    # Replicated (non-FSDP) parameter gradients must be identical across ranks.
    for name, param in fsdp_model.module.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.rsplit(".", 1)
        modules = dict(fsdp_model.module.named_modules())
        mod = modules[parts[0]] if len(parts) == 2 else fsdp_model.module
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        gathered = [torch.zeros_like(param.grad) for _ in range(world_size)]
        dist.all_gather(gathered, param.grad)
        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4, rtol=1e-4), (
                f"Replicated gradient for {name} differs between rank 0 and rank {r}. "
                f"Max diff: {(gathered[0] - gathered[r]).abs().max().item()}"
            )

    if rank == 0:
        tag = "fp32" if compute_dtype is None else str(compute_dtype)
        print(f"test_fsdp_gradient_sync[{tag}]: grads shaped/typed and replicas synced.")

    _cleanup_process_group()


# ---------------------------------------------------------------------------
# Robustness: activation-dtype handling for arbitrary (fp32-activation) models
# ---------------------------------------------------------------------------


def _test_fsdp_activation_dtype(rank: int, world_size: int, compute_dtype) -> None:
    """FSDP must run a model whose activations are NOT already in compute_dtype
    (here, a Linear-first MLP fed fp32 input) and match a single-process model
    with matching boundary mixed-precision hooks. Also checks masters/grads
    stay fp32."""
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(7)
    base_model = ToyMLPModel(d_in=16, d_hidden=32, d_out=16).to(device)

    ref_model = deepcopy(base_model)
    if compute_dtype is not None:
        _apply_boundary_mixed_precision_hooks(ref_model, compute_dtype)

    fsdp_model = get_fsdp(deepcopy(base_model), compute_dtype=compute_dtype)

    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.05)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=0.05)

    torch.manual_seed(99)
    batch_size = 16
    all_x = torch.randn(batch_size, 16, device=device)
    all_y = torch.randn(batch_size, 16, device=device)
    local_bs = batch_size // world_size

    for step in range(3):
        fsdp_optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)

        ref_out = ref_model(all_x)
        ref_loss = F.mse_loss(ref_out.float(), all_y)
        ref_loss.backward()
        ref_optimizer.step()

        offset = rank * local_bs
        local_x = all_x[offset : offset + local_bs]
        local_y = all_y[offset : offset + local_bs]
        fsdp_out = fsdp_model(local_x)
        fsdp_loss = F.mse_loss(fsdp_out.float(), local_y)
        fsdp_loss.backward()

        fsdp_on_after_backward(fsdp_model, fsdp_optimizer)
        fsdp_optimizer.step()

        # Master weights and their grads must stay fp32 regardless of compute_dtype.
        for name, param in fsdp_model.module.named_parameters():
            assert param.data.dtype == torch.float32, f"{name} master is {param.data.dtype}"
            if param.grad is not None:
                assert param.grad.dtype == torch.float32, f"{name} grad is {param.grad.dtype}"

        full_params = fsdp_gather_full_params(fsdp_model)
        atol = 1e-6 if compute_dtype is None else 2e-3
        rtol = 1e-4 if compute_dtype is None else 2e-3
        for name, ref_param in ref_model.named_parameters():
            fsdp_full = full_params[name]
            assert torch.allclose(ref_param.data, fsdp_full, atol=atol, rtol=rtol), (
                f"Step {step}: Parameter {name} mismatch. Max diff: "
                f"{(ref_param.data - fsdp_full).abs().max().item()}"
            )

    if rank == 0:
        tag = "fp32" if compute_dtype is None else str(compute_dtype)
        print(f"test_fsdp_activation_dtype[{tag}]: fp32-activation MLP ran and matched reference.")

    _cleanup_process_group()


# ---------------------------------------------------------------------------
# Frozen sharded params: requires_grad must survive FSDP wrapping
# ---------------------------------------------------------------------------


def _test_fsdp_preserves_requires_grad(rank: int, world_size: int, compute_dtype) -> None:
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(17)
    model = FrozenShardedParamModel().to(device)
    fsdp_model = get_fsdp(deepcopy(model), compute_dtype=compute_dtype)

    frozen_names = FrozenShardedParamModel.frozen_names
    for name, param in fsdp_model.module.named_parameters():
        assert param.requires_grad is (name not in frozen_names), (
            f"{name} requires_grad={param.requires_grad}, expected {name not in frozen_names}"
        )

    before = fsdp_gather_full_params(fsdp_model)
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.05)

    torch.manual_seed(200 + rank)
    input_ids = torch.randint(0, 29, (6, 4), device=device)
    target = torch.randn(6, 5, device=device)

    out = fsdp_model(input_ids)
    loss = F.mse_loss(out.float(), target)
    loss.backward()
    fsdp_model.finish_gradient_synchronization()

    for name, param in fsdp_model.module.named_parameters():
        if name in frozen_names:
            assert param.grad is None, f"Frozen sharded parameter {name} received a gradient"
        else:
            assert param.grad is not None, f"Trainable sharded parameter {name} did not receive a gradient"
            assert param.grad.shape == param.data.shape
            assert param.grad.dtype == param.data.dtype

    optimizer.step()
    after = fsdp_gather_full_params(fsdp_model)

    for name in frozen_names:
        assert torch.equal(before[name], after[name]), f"Frozen sharded parameter {name} changed"

    if rank == 0:
        tag = "fp32" if compute_dtype is None else str(compute_dtype)
        print(f"test_fsdp_preserves_requires_grad[{tag}]: frozen sharded params stayed frozen.")

    _cleanup_process_group()


# ---------------------------------------------------------------------------
# Runners (script equivalents of the pytest-parametrized entry points)
# ---------------------------------------------------------------------------

WORLD_SIZE = 2
_PORT = 12357


def _spawn(fn, compute_dtype) -> None:
    global _PORT
    _PORT += 1
    os.environ["MASTER_PORT"] = str(_PORT)
    mp.spawn(fn, args=(WORLD_SIZE, compute_dtype), nprocs=WORLD_SIZE, join=True)


def test_fsdp_correctness(compute_dtype) -> None:
    _spawn(_test_fsdp_correctness, compute_dtype)


def test_fsdp_gradient_sync(compute_dtype) -> None:
    _spawn(_test_fsdp_gradient_sync, compute_dtype)


def test_fsdp_activation_dtype(compute_dtype) -> None:
    _spawn(_test_fsdp_activation_dtype, compute_dtype)


def test_fsdp_preserves_requires_grad(compute_dtype) -> None:
    _spawn(_test_fsdp_preserves_requires_grad, compute_dtype)


def main() -> None:
    for compute_dtype in (None, torch.float16):
        test_fsdp_correctness(compute_dtype)
        test_fsdp_gradient_sync(compute_dtype)
        test_fsdp_activation_dtype(compute_dtype)
        test_fsdp_preserves_requires_grad(compute_dtype)


if __name__ == "__main__":
    main()
