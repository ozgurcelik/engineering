"""Profile FSDP vs a full-replica baseline on a toy Transformer LM (small GPT),
sweeping the SEQUENCE LENGTH.

Why sweep sequence length?
  Parameters and optimizer state do NOT depend on sequence length, so FSDP's
  sharding win on them is a fixed number of bytes. Attention activations, on the
  other hand, grow ~O(T^2) (this toy uses explicit softmax(QK^T)V so the
  [B, heads, T, T] scores are materialized and saved for backward). As T grows,
  activations come to dominate total memory, and since activations are the SAME
  on FSDP and baseline (FSDP shards weights, not activations), FSDP's relative
  advantage shrinks. This sweep makes that trade-off visible.

We reuse the model-agnostic measurement helpers from ``profile_fsdp.py``:
  * peak process RSS (everything: live tensors + allocator caching + transient),
  * the profiler's per-category LIVE tensor co-peak (params/optimizer/grad/
    activation/autograd/...), which is the "true" live-memory view.

Run (CPU / gloo):

    conda run -n distvenv python parallelism/profile_fsdp_transformers.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from fsdp import FSDP
from profile_fsdp import (
    MB,
    CATEGORY_ORDER,
    FullWeightTracker,
    PeakRSS,
    _categorized_memory,
    _footprint_bytes,
    _record_fsdp_internal_sample,
    _resolve_device_type,
    _report_categories,
    _report_fsdp_internal,
    _write_memory_timeline_html,
)

WORLD_SIZE = 2
STEPS = 3

# Toy GPT config. Kept small so a CPU/gloo sweep across sequence lengths runs in
# a couple of minutes, but with enough params that the sharding win is visible.
VOCAB = 4096
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
D_FF = 4 * D_MODEL
BATCH_PER_RANK = 4

SEQ_LENS = [128, 256, 512, 1024]
MAX_SEQ = max(SEQ_LENS)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "transformer"
)


# ---------------------------------------------------------------------------
# Toy causal Transformer LM (a small GPT). Only nn.Linear / nn.Embedding are
# sharded by FSDP; LayerNorm weights/biases stay replicated.
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with EXPLICIT scores (softmax(QK^T)V).

    We deliberately materialize the [B, heads, T, T] score/prob tensors (instead
    of a fused/flash kernel) so the O(T^2) activation growth is visible in the
    memory profile as sequence length increases."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [b, h, t, t]
        causal = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        y = attn @ v  # [b, h, t, d_head]
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class ToyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab: int = VOCAB,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        max_seq: int = MAX_SEQ,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Process group + training helpers
# ---------------------------------------------------------------------------


def _setup(rank: int, world_size: int, device_type: str = "cpu") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "12533")
    if device_type == "cuda":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


def _make_batch(seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, VOCAB, (BATCH_PER_RANK, seq_len), device=device)
    y = torch.randint(0, VOCAB, (BATCH_PER_RANK, seq_len), device=device)
    return x, y


def _train_step(
    model: nn.Module,
    optimizer,
    is_fsdp: bool,
    x,
    y,
    fsdp_samples: list[dict[str, int | str]] | None = None,
    tracker: FullWeightTracker | None = None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    _record_fsdp_internal_sample(fsdp_samples, "after_zero_grad", model, tracker)
    with record_function("forward"):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), y.reshape(-1))
    _record_fsdp_internal_sample(fsdp_samples, "after_forward", model, tracker)
    with record_function("backward"):
        loss.backward()
    _record_fsdp_internal_sample(fsdp_samples, "after_backward_before_finish", model, tracker)
    if is_fsdp:
        with record_function("finish_grad_sync"):
            model.finish_gradient_synchronization()
        _record_fsdp_internal_sample(fsdp_samples, "after_finish_grad_sync", model, tracker)
    with record_function("optimizer_step"):
        optimizer.step()
    _record_fsdp_internal_sample(fsdp_samples, "after_optimizer_step", model, tracker)


def run_phase(
    rank: int, mode: str, world_size: int, seq_len: int, device_type: str = "cpu"
) -> None:
    proc = psutil.Process()
    if device_type == "cuda":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if mode == "fsdp":
        _setup(rank, world_size, device_type)

    # Capture the pre-model baseline AFTER comm init so comm overhead is not
    # charged to model/training memory (fair across modes). On CUDA the analogue
    # is the allocator's peak-allocated high-water mark (context excluded).
    rss_before = proc.memory_info().rss

    torch.manual_seed(0)
    if mode == "fsdp":
        model: nn.Module = FSDP(ToyTransformerLM().to(device), compute_dtype=None)
        is_fsdp = True
    else:
        model = ToyTransformerLM().to(device)
        is_fsdp = False

    # Tracks full-weight storages pinned alive by autograd until backward, which
    # a `full_param.data` scan misses (same instrument the other profiler uses).
    tracker = FullWeightTracker(model) if mode == "fsdp" and rank == 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = _make_batch(seq_len, device)

    # --- Pass 1: clean peak-memory run (no profiler) ---
    footprint: dict[str, int] = {}
    fsdp_internal_samples: list[dict[str, int | str]] = []

    def _run_pass1() -> None:
        nonlocal footprint
        for step in range(STEPS):
            samples = (
                fsdp_internal_samples
                if mode == "fsdp" and rank == 0 and step == 0
                else None
            )
            with record_function("step"):
                _train_step(model, optimizer, is_fsdp, x, y, samples, tracker)
            if step == 0:
                footprint = _footprint_bytes(model, optimizer)

    if device_type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        _run_pass1()
        torch.cuda.synchronize(device)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_rss = 0
    else:
        with PeakRSS() as peak:
            _run_pass1()
        peak_rss = peak.peak
        peak_alloc = 0
        peak_reserved = 0

    # --- Pass 2: profiler run for the per-category live-memory breakdown ---
    activities = [ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(STEPS):
            with record_function("step"):
                _train_step(model, optimizer, is_fsdp, x, y)

    memprof = prof._memory_profile()
    cat_device = str(device) if device_type == "cuda" else None
    categorized = _categorized_memory(memprof, cat_device)

    top_ops = ""
    if rank == 0:
        sort_key = "self_cuda_memory_usage" if device_type == "cuda" else "self_cpu_memory_usage"
        top_ops = prof.key_averages().table(sort_by=sort_key, row_limit=10)
        _write_memory_timeline_html(
            os.path.join(RESULTS_DIR, f"{mode}_seq{seq_len}_rank0_memory.html"),
            f"{mode} seq={seq_len}",
            memprof,
            cat_device,
        )

    result = {
        "mode": mode,
        "rank": rank,
        "seq_len": seq_len,
        "device_type": device_type,
        "rss_before": rss_before,
        "peak_rss": peak_rss,
        "peak_cuda_alloc": peak_alloc,
        "peak_cuda_reserved": peak_reserved,
        **footprint,
        **categorized,
        "fsdp_internal_samples": fsdp_internal_samples,
        "top_ops": top_ops,
    }
    with open(os.path.join(RESULTS_DIR, f"{mode}_seq{seq_len}_{rank}.json"), "w") as f:
        json.dump(result, f)

    if mode == "fsdp":
        _cleanup()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(b: float) -> str:
    return f"{b / MB:8.1f}"


def _load_results() -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            row = json.load(f)
        if "mode" not in row:
            continue
        # rank 0 is representative for the per-category / footprint numbers.
        if row.get("rank", 0) != 0:
            continue
        rows[(row["mode"], row["seq_len"])] = row
    return rows


def _report(seq_lens: list[int]) -> None:
    rows = _load_results()

    with torch.device("meta"):
        n_params = sum(p.numel() for p in ToyTransformerLM().parameters())

    print("\n" + "#" * 92)
    print(
        f"# TOY TRANSFORMER LM — SEQUENCE-LENGTH SWEEP  (world_size={WORLD_SIZE}, "
        f"{STEPS} steps, batch/rank={BATCH_PER_RANK})"
    )
    print(
        f"# {N_LAYERS} layers, d_model={D_MODEL}, heads={N_HEADS}, d_ff={D_FF}, "
        f"vocab={VOCAB}, params={n_params / 1e6:.1f}M"
    )
    print("#" * 92)

    # Seq-independent footprint (params/optimizer are constant in T); print once.
    base0 = rows.get(("baseline", seq_lens[0]))
    fsdp0 = rows.get(("fsdp", seq_lens[0]))
    if base0 and fsdp0:
        print(
            f"resident params+grad+opt (T-independent):  baseline {_fmt(base0['resident_total'])} MB"
            f"   fsdp/rank {_fmt(fsdp0['resident_total'])} MB"
            f"   ({base0['resident_total'] / max(fsdp0['resident_total'], 1):.2f}x)"
        )

    # --- Table 1: LIVE co-peak tensor memory (the true memory picture) ---
    print("\n" + "=" * 92)
    print("LIVE co-peak tensor memory vs sequence length (MB)   [profiler: sum of live tensors]")
    print("=" * 92)
    header = (
        f"{'seq_len':>8}{'acts base':>12}{'acts fsdp':>12}"
        f"{'TOTAL base':>13}{'TOTAL fsdp':>13}{'TOTAL ratio':>13}"
    )
    print(header)
    print("-" * len(header))
    for t in seq_lens:
        b = rows.get(("baseline", t))
        f = rows.get(("fsdp", t))
        if not b or not f:
            continue
        ratio = b["cat_TOTAL"] / max(f["cat_TOTAL"], 1)
        print(
            f"{t:>8}{_fmt(b['cat_ACTIVATION']):>12}{_fmt(f['cat_ACTIVATION']):>12}"
            f"{_fmt(b['cat_TOTAL']):>13}{_fmt(f['cat_TOTAL']):>13}{ratio:>11.2f}x"
        )
    print("-" * len(header))
    print(
        "note: params/optimizer/grad are ~constant in T and sharded ~world_size by\n"
        "      FSDP; activations grow ~O(T^2) and are IDENTICAL on baseline vs fsdp\n"
        "      (FSDP shards weights, not activations). So as T grows, activations\n"
        "      dominate and the FSDP TOTAL ratio decays toward 1.0."
    )

    # --- Table 2: per-category live co-peak for FSDP across T (where bytes go) ---
    cats = ["PARAMETER", "OPTIMIZER_STATE", "GRADIENT", "ACTIVATION", "AUTOGRAD_DETAIL"]
    print("\n" + "=" * 92)
    print("FSDP/rank live co-peak by category vs sequence length (MB)")
    print("=" * 92)
    header = f"{'seq_len':>8}" + "".join(f"{c.split('_')[0].lower():>16}" for c in cats)
    print(header)
    print("-" * len(header))
    for t in seq_lens:
        f = rows.get(("fsdp", t))
        if not f:
            continue
        row = f"{t:>8}"
        for c in cats:
            row += f"{_fmt(f[f'cat_{c}']):>16}"
        print(row)

    # --- Table 3: peak process memory (everything, incl. allocator churn) ---
    device_type = (fsdp0 or base0 or {}).get("device_type", "cpu")
    is_cuda = device_type == "cuda"

    def _peak_above_overhead(r: dict) -> int:
        # CUDA: max_memory_allocated already excludes the fixed context. CPU:
        # subtract the pre-model RSS overhead.
        return r["peak_cuda_alloc"] if is_cuda else r["peak_rss"] - r["rss_before"]

    print("\n" + "=" * 92)
    if is_cuda:
        print("Peak CUDA memory allocated vs sequence length (MB)   [torch.cuda.max_memory_allocated]")
    else:
        print("Peak process RSS above overhead vs sequence length (MB)   [OS resident, all memory]")
    print("=" * 92)
    col = "CUDA base" if is_cuda else "RSS base"
    colf = "CUDA fsdp" if is_cuda else "RSS fsdp"
    header = f"{'seq_len':>8}{col:>13}{colf:>13}{'ratio':>13}"
    print(header)
    print("-" * len(header))
    for t in seq_lens:
        b = rows.get(("baseline", t))
        f = rows.get(("fsdp", t))
        if not b or not f:
            continue
        bo = _peak_above_overhead(b)
        fo = _peak_above_overhead(f)
        print(
            f"{t:>8}{_fmt(bo):>13}{_fmt(fo):>13}{bo / max(fo, 1):>11.2f}x"
        )
    print("-" * len(header))
    if is_cuda:
        print(
            "note: max_memory_allocated is the high-water mark of LIVE device tensors\n"
            "      (excludes the fixed CUDA context and the allocator's reserved-but-\n"
            "      unused blocks), so it tracks the live-tensor co-peak closely."
        )
    else:
        print(
            "note: RSS includes live tensors + CPU-allocator caching/fragmentation +\n"
            "      transient spikes, so it tracks the live-tensor ratio only loosely\n"
            "      (FSDP's gather/reduce-scatter/resize churn inflates it)."
        )


def _report_detail(seq_len: int) -> None:
    """Fine-grained, single-seq-length view — the same breakdown the plain
    ``profile_fsdp.py`` prints: full per-category peak tensor memory, the
    FSDP-internal transient-buffer trace through one step, and the top ops by
    self CPU memory. Focused on one T so the numbers are readable."""
    rows = _load_results()
    base = rows.get(("baseline", seq_len))
    fsdp0 = rows.get(("fsdp", seq_len))

    print("\n" + "#" * 92)
    print(f"# FINE-GRAINED DETAIL @ seq_len={seq_len}  (fsdp = rank 0, world_size={WORLD_SIZE})")
    print("#" * 92)

    # Reuse the exact reporting the base profiler uses (dict-driven, model-agnostic).
    _report_categories(base, fsdp0)
    _report_fsdp_internal(fsdp0)

    for r, tag in ((base, "BASELINE"), (fsdp0, "FSDP")):
        if not r or not r.get("top_ops"):
            continue
        print("\n" + "=" * 92)
        print(f"{tag} (rank 0) @ seq_len={seq_len} — top ops by self CPU memory (transient)")
        print("=" * 92)
        print(r["top_ops"])


def main(device: str = "cpu", seq_lens: list[int] | None = None) -> None:
    device_type = _resolve_device_type(device)
    requested = seq_lens or SEQ_LENS

    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for seq_len in requested:
        # Separate processes per phase so each gets a clean peak-memory mark.
        mp.spawn(run_phase, args=("baseline", 1, seq_len, device_type), nprocs=1, join=True)
        mp.spawn(
            run_phase, args=("fsdp", WORLD_SIZE, seq_len, device_type), nprocs=WORLD_SIZE, join=True
        )

    if len(requested) > 1:
        _report(requested)
    # Always print the fine-grained single-T detail for the largest T run.
    _report_detail(max(requested))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="cpu (gloo, default) or cuda (NCCL, one rank per GPU).",
    )
    parser.add_argument(
        "seq_lens",
        nargs="*",
        type=int,
        help="Optional sequence lengths to run (default: the built-in sweep).",
    )
    args = parser.parse_args()
    main(device=args.device, seq_lens=args.seq_lens or None)
