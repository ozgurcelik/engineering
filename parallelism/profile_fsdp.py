"""Compare TOTAL training memory of FSDP (world_size=2) vs a full-replica baseline.

"Total" here means peak process RSS during the training loop, which includes
*everything*: parameters, gradients, optimizer state, activations, autograd
saved tensors, and FSDP's transient all-gather / reduce-scatter buffers.

Why peak RSS (and not just the profiler table)?
  * On CPU there is no ``torch.cuda.max_memory_allocated``.
  * Activations are allocated and freed within a step, so we sample *current*
    RSS on a background thread and keep the max.
  * RSS high-water marks don't reset within a process and the CPU allocator
    caches freed blocks, so baseline and FSDP are each run in their OWN process
    to get clean, comparable peaks.

We also keep the PyTorch profiler memory table from
https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
for the per-op transient-allocation breakdown.

Run (CPU / gloo):

    conda run -n distvenv python parallelism/profile_fsdp.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import threading
import time
from collections import defaultdict

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from fsdp import FSDP

WORLD_SIZE = 2
STEPS = 5

VOCAB = 2000
D_MODEL = 1024
D_FF = 4096
NUM_LAYERS = 24
# Small batch/seq so activations don't dominate — this is the regime where the
# parameter/optimizer-state sharding win drives total memory down.
BATCH_PER_RANK = 8
SEQ_LEN = 32
MB = 1024 * 1024

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Order (and friendly labels) for the granular per-category memory breakdown.
# These are exactly the buckets torch's memory profiler assigns every live
# allocation to, reconstructed from the recorded op + autograd graph.
CATEGORY_ORDER = [
    ("PARAMETER", "parameters"),
    ("OPTIMIZER_STATE", "optimizer state"),
    ("GRADIENT", "gradients"),
    ("ACTIVATION", "activations"),
    ("AUTOGRAD_DETAIL", "autograd saved"),
    ("TEMPORARY", "temporaries"),
    ("INPUT", "inputs"),
    ("UNKNOWN", "uncategorized"),
]

CATEGORY_COLORS = {
    "PARAMETER": "#1b7f3a",
    "OPTIMIZER_STATE": "#c89c1f",
    "GRADIENT": "#2453d4",
    "ACTIVATION": "#d94b45",
    "AUTOGRAD_DETAIL": "#4878c8",
    "TEMPORARY": "#8b63c7",
    "INPUT": "#30343b",
    "UNKNOWN": "#9a9a9a",
}


def _categorized_memory(memprof) -> dict[str, int]:
    """Walk the profiler's memory timeline and return the PEAK live bytes per
    category (PARAMETER / OPTIMIZER_STATE / GRADIENT / ACTIVATION /
    AUTOGRAD_DETAIL / TEMPORARY / INPUT / UNKNOWN), plus a co-peak TOTAL.

    torch categorizes every allocation from the recorded op + autograd graph, so
    this splits the coarse RSS number into where the bytes actually live. Each
    category peak is its own high-water mark; ``TOTAL`` is the max of the *sum*
    of all live tensors at a single instant (they don't all peak together)."""
    # Track current size per storage key so in-place version bumps don't double
    # count. key -> (nbytes, category_name).
    live: dict[object, tuple[int, str]] = {}
    cat_peak: dict[str, int] = defaultdict(int)
    total_peak = 0

    for _t, action, key_ver, nbytes in memprof.timeline:
        key, version = key_ver
        cat = memprof._categories.get(key, version)
        cname = cat.name if cat is not None else "UNKNOWN"
        a = action.name
        if a in ("PREEXISTING", "CREATE", "INCREMENT_VERSION"):
            live[key] = (nbytes, cname)
        elif a == "DESTROY":
            live.pop(key, None)

        sums: dict[str, int] = defaultdict(int)
        for b, c in live.values():
            sums[c] += b
        for c, v in sums.items():
            if v > cat_peak[c]:
                cat_peak[c] = v
        total = sum(sums.values())
        if total > total_peak:
            total_peak = total

    result = {f"cat_{name}": cat_peak.get(name, 0) for name, _ in CATEGORY_ORDER}
    result["cat_TOTAL"] = total_peak
    return result


def _memory_timeline_points(memprof) -> dict[str, object]:
    """Build browser-friendly timeline points from the profiler memory timeline."""
    live: dict[object, tuple[int, str]] = {}
    category_names = [name for name, _ in CATEGORY_ORDER]
    start_time = next((t for t, *_ in memprof.timeline if t >= 0), 0)
    points: list[list[float]] = []
    last_values: list[float] | None = None

    for t, action, key_ver, nbytes in memprof.timeline:
        key, version = key_ver
        cat = memprof._categories.get(key, version)
        cname = cat.name if cat is not None else "UNKNOWN"
        a = action.name
        if a in ("PREEXISTING", "CREATE", "INCREMENT_VERSION"):
            live[key] = (nbytes, cname)
        elif a == "DESTROY":
            live.pop(key, None)

        sums: dict[str, int] = defaultdict(int)
        for b, c in live.values():
            sums[c] += b

        values = [sums.get(name, 0) / MB for name in category_names]
        values.append(sum(sums.values()) / MB)
        rel_ms = 0.0 if t < 0 else (t - start_time) / 1_000_000

        if values != last_values:
            points.append([rel_ms, *values])
            last_values = values

    return {
        "categories": category_names,
        "labels": [label for _, label in CATEGORY_ORDER],
        "colors": [CATEGORY_COLORS[name] for name in category_names],
        "points": points,
    }


def _write_memory_timeline_html(path: str, mode: str, memprof) -> None:
    data = _memory_timeline_points(memprof)
    data["mode"] = mode
    payload = json.dumps(data)
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FSDP memory timeline</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; background: #fafafa; }
    main { padding: 24px; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    p { margin: 0 0 18px; color: #59636e; }
    #wrap { position: relative; height: 620px; background: white; border: 1px solid #d8dee4; border-radius: 8px; }
    canvas { width: 100%; height: 100%; display: block; }
    #legend { display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 14px; font-size: 13px; }
    .item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    #tip { position: fixed; pointer-events: none; display: none; padding: 8px 10px; border: 1px solid #d8dee4; background: rgba(255, 255, 255, 0.96); border-radius: 6px; font-size: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.12); white-space: nowrap; }
    #tip b { display: block; margin-bottom: 4px; }
  </style>
</head>
<body>
  <main>
    <h1>Memory Timeline: <span id="mode"></span> rank 0</h1>
    <p>Stacked live tensor memory by PyTorch profiler category. Values are MiB.</p>
    <div id="wrap"><canvas id="chart"></canvas></div>
    <div id="legend"></div>
  </main>
  <div id="tip"></div>
  <script>
    const DATA = __DATA__;
    document.getElementById("mode").textContent = DATA.mode;
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const tip = document.getElementById("tip");
    const legend = document.getElementById("legend");
    const points = DATA.points;
    const n = DATA.categories.length;
    const margin = { left: 72, right: 22, top: 24, bottom: 48 };
    const maxT = Math.max(...points.map(p => p[0]), 1);
    const maxY = Math.max(...points.map(p => p[n + 1]), 1);

    DATA.labels.forEach((label, i) => {
      const el = document.createElement("span");
      el.className = "item";
      el.innerHTML = `<span class="swatch" style="background:${DATA.colors[i]}"></span>${label}`;
      legend.appendChild(el);
    });

    function xFor(t, w) {
      return margin.left + (t / maxT) * (w - margin.left - margin.right);
    }

    function yFor(v, h) {
      return h - margin.bottom - (v / maxY) * (h - margin.top - margin.bottom);
    }

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = "#d8dee4";
      ctx.fillStyle = "#59636e";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 5; i++) {
        const v = (maxY * i) / 5;
        const y = yFor(v, h);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(w - margin.right, y);
        ctx.stroke();
        ctx.fillText(v.toFixed(0), margin.left - 8, y);
      }

      let cumulative = new Array(points.length).fill(0);
      for (let c = 0; c < n; c++) {
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const x = xFor(points[i][0], w);
          const y = yFor(cumulative[i] + points[i][c + 1], h);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        for (let i = points.length - 1; i >= 0; i--) {
          const x = xFor(points[i][0], w);
          const y = yFor(cumulative[i], h);
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.globalAlpha = 0.68;
        ctx.fillStyle = DATA.colors[c];
        ctx.fill();
        ctx.globalAlpha = 1;
        for (let i = 0; i < points.length; i++) cumulative[i] += points[i][c + 1];
      }

      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const x = xFor(points[i][0], w);
        const y = yFor(points[i][n + 1], h);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = "#59636e";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText("time (ms)", (margin.left + w - margin.right) / 2, h - 30);
      ctx.save();
      ctx.translate(20, (margin.top + h - margin.bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("MiB", 0, 0);
      ctx.restore();
    }

    function nearestPoint(clientX) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const t = Math.max(0, Math.min(maxT, ((x - margin.left) / (rect.width - margin.left - margin.right)) * maxT));
      let lo = 0, hi = points.length - 1;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (points[mid][0] < t) lo = mid + 1; else hi = mid;
      }
      const a = Math.max(0, lo - 1);
      const b = lo;
      return Math.abs(points[a][0] - t) < Math.abs(points[b][0] - t) ? points[a] : points[b];
    }

    canvas.addEventListener("mousemove", event => {
      const p = nearestPoint(event.clientX);
      const rows = DATA.labels.map((label, i) => `${label}: ${p[i + 1].toFixed(1)} MiB`);
      tip.innerHTML = `<b>${p[0].toFixed(2)} ms, total ${p[n + 1].toFixed(1)} MiB</b>${rows.join("<br>")}`;
      tip.style.left = `${event.clientX + 14}px`;
      tip.style.top = `${event.clientY + 14}px`;
      tip.style.display = "block";
    });
    canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    window.addEventListener("resize", draw);
    draw();
  </script>
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(html.replace("__DATA__", payload))


class Block(nn.Module):
    """A residual MLP block: two Linears (sharded by FSDP) + a replicated norm."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.relu(self.linear1(h))
        h = self.linear2(h)
        return x + h


class ToyModel(nn.Module):
    """Embedding + a deep stack of residual MLP blocks + head. With NUM_LAYERS
    blocks there are 2*NUM_LAYERS + 2 FSDP-sharded layers (Embedding/Linears),
    so parameter + optimizer-state memory dominates."""

    def __init__(
        self,
        vocab: int = VOCAB,
        d_model: int = D_MODEL,
        d_ff: int = D_FF,
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class PeakRSS:
    """Sample current process RSS on a background thread and keep the max.
    Captures transient activation memory that alloc/frees within a step."""

    def __init__(self, interval: float = 0.0005) -> None:
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRSS":
        self.peak = self.proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "12413")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


def _footprint_bytes(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Resident bytes held on THIS rank: parameters, grads, optimizer state."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    opt_bytes = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                opt_bytes += v.numel() * v.element_size()
    return {
        "params": param_bytes,
        "grads": grad_bytes,
        "opt_state": opt_bytes,
        "resident_total": param_bytes + grad_bytes + opt_bytes,
    }


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _fsdp_internal_breakdown(model: nn.Module) -> dict[str, int]:
    """Semantic memory counters from this learning FSDP's own state.

    PyTorch's profiler categories are useful but generic; they infer labels from
    autograd/dataflow. These counters name the FSDP-specific transient buffers.
    """
    if not isinstance(model, FSDP):
        return {}

    local_shard_params = 0
    local_shard_grads = 0
    forward_all_gather = 0
    backward_all_gather = 0
    active_full_params = 0

    for layer_state in model._layer_states.values():
        for param_state in layer_state.param_states.values():
            local_shard_params += _tensor_bytes(param_state.local_param.data)
            local_shard_grads += _tensor_bytes(param_state.local_param.grad)

            full_param = param_state.full_param
            if full_param is None:
                continue
            full_bytes = _tensor_bytes(full_param.data)
            if param_state.forward_gather_handle is not None:
                forward_all_gather += full_bytes
            elif param_state.backward_gather_handle is not None:
                backward_all_gather += full_bytes
            else:
                active_full_params += full_bytes

    pending_reduce_scatter_input = sum(
        _tensor_bytes(pending.input_keepalive)
        for pending in model._pending_reduce_scatters
    )
    pending_reduce_scatter_output = sum(
        _tensor_bytes(pending.output)
        for pending in model._pending_reduce_scatters
    )

    return {
        "local_shard_params": local_shard_params,
        "local_shard_grads": local_shard_grads,
        "forward_all_gather_buffers": forward_all_gather,
        "backward_all_gather_buffers": backward_all_gather,
        "active_full_params": active_full_params,
        "pending_reduce_scatter_full_grad_inputs": pending_reduce_scatter_input,
        "pending_reduce_scatter_shard_outputs": pending_reduce_scatter_output,
        "tracked_total": (
            local_shard_params
            + local_shard_grads
            + forward_all_gather
            + backward_all_gather
            + active_full_params
            + pending_reduce_scatter_input
            + pending_reduce_scatter_output
        ),
    }


def _record_fsdp_internal_sample(
    samples: list[dict[str, int | str]] | None,
    label: str,
    model: nn.Module,
) -> None:
    if samples is None:
        return
    breakdown = _fsdp_internal_breakdown(model)
    if breakdown:
        samples.append({"point": label, **breakdown})


def _make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, VOCAB, (BATCH_PER_RANK, SEQ_LEN), device=device)
    y = torch.randint(0, VOCAB, (BATCH_PER_RANK, SEQ_LEN), device=device)
    return x, y


def _train_step(
    model: nn.Module,
    optimizer,
    is_fsdp: bool,
    x,
    y,
    fsdp_samples: list[dict[str, int | str]] | None = None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    _record_fsdp_internal_sample(fsdp_samples, "after_zero_grad", model)
    with record_function("forward"):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), y.reshape(-1))
    _record_fsdp_internal_sample(fsdp_samples, "after_forward", model)
    with record_function("backward"):
        loss.backward()
    _record_fsdp_internal_sample(fsdp_samples, "after_backward_before_finish", model)
    if is_fsdp:
        with record_function("finish_grad_sync"):
            model.finish_gradient_synchronization()
        _record_fsdp_internal_sample(fsdp_samples, "after_finish_grad_sync", model)
    with record_function("optimizer_step"):
        optimizer.step()
    _record_fsdp_internal_sample(fsdp_samples, "after_optimizer_step", model)


def run_phase(rank: int, mode: str, world_size: int) -> None:
    proc = psutil.Process()
    rss_before = proc.memory_info().rss  # interpreter + torch import overhead
    device = torch.device("cpu")

    if mode == "fsdp":
        _setup(rank, world_size)

    torch.manual_seed(0)
    if mode == "fsdp":
        model: nn.Module = FSDP(ToyModel().to(device), compute_dtype=None)
        is_fsdp = True
    else:
        model = ToyModel().to(device)
        is_fsdp = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = _make_batch(device)

    # --- Pass 1: clean peak-RSS run (no profiler, to avoid contaminating RSS) ---
    footprint: dict[str, int] = {}
    fsdp_internal_samples: list[dict[str, int | str]] = []
    with PeakRSS() as peak:
        for step in range(STEPS):
            samples = fsdp_internal_samples if mode == "fsdp" and rank == 0 and step == 0 else None
            with record_function("step"):
                _train_step(model, optimizer, is_fsdp, x, y, samples)
            if step == 0:
                footprint = _footprint_bytes(model, optimizer)
    peak_rss = peak.peak

    # --- Pass 2: profiler run for the granular per-category memory breakdown ---
    # record_shapes + with_stack are REQUIRED for the memory profiler to walk the
    # op/autograd graph and label allocations (activation vs grad vs autograd ...).
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(STEPS):
            with record_function("step"):
                _train_step(model, optimizer, is_fsdp, x, y)

    memprof = prof._memory_profile()
    categorized = _categorized_memory(memprof)

    if rank == 0:
        prof.export_chrome_trace(os.path.join(RESULTS_DIR, f"{mode}_rank0_trace.json"))
        _write_memory_timeline_html(
            os.path.join(RESULTS_DIR, f"{mode}_rank0_memory.html"),
            mode,
            memprof,
        )
        prof.export_memory_timeline(
            os.path.join(RESULTS_DIR, f"{mode}_rank0_memory.raw.json.gz"),
            device="cpu",
        )

    result = {
        "mode": mode,
        "rank": rank,
        "rss_before": rss_before,
        "peak_rss": peak_rss,
        **footprint,
        **categorized,
        "fsdp_internal_samples": fsdp_internal_samples,
    }
    with open(os.path.join(RESULTS_DIR, f"{mode}_{rank}.json"), "w") as f:
        json.dump(result, f)

    if rank == 0:
        tag = f"{mode.upper()} (rank {rank})"
        print(f"\n{'=' * 78}\n{tag} — top ops by self CPU memory (transient)\n{'=' * 78}")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=8))

    if mode == "fsdp":
        _cleanup()


def _fmt(b: float) -> str:
    return f"{b / MB:8.2f} MB"


def _report() -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        filename = os.path.basename(path)
        if filename.endswith("_trace.json") or "_memory" in filename:
            continue
        with open(path) as f:
            row = json.load(f)
        if "mode" in row:
            rows.append(row)

    base = next((r for r in rows if r["mode"] == "baseline"), None)
    fsdp0 = next((r for r in rows if r["mode"] == "fsdp" and r["rank"] == 0), None)

    with torch.device("meta"):
        n_params = sum(p.numel() for p in ToyModel().parameters())
    print("\n" + "#" * 78)
    print(f"# TOTAL TRAINING MEMORY  (world_size={WORLD_SIZE}, {STEPS} steps, batch/rank={BATCH_PER_RANK})")
    print(f"# model: {NUM_LAYERS} blocks, d_model={D_MODEL}, d_ff={D_FF}, params={n_params / 1e6:.1f}M")
    print("#" * 78)
    header = f"{'phase':<20}{'peak RSS':>14}{'peak-overhead':>16}{'resident(P+G+O)':>18}{'activations~':>16}"
    print(header)
    print("-" * len(header))
    for r in (base, fsdp0):
        if r is None:
            continue
        peak_minus_overhead = r["peak_rss"] - r["rss_before"]
        activations = peak_minus_overhead - r["resident_total"]
        label = "baseline (full)" if r["mode"] == "baseline" else "fsdp (per rank)"
        print(
            f"{label:<20}{_fmt(r['peak_rss']):>14}{_fmt(peak_minus_overhead):>16}"
            f"{_fmt(r['resident_total']):>18}{_fmt(activations):>16}"
        )

    if base and fsdp0:
        bd = base["peak_rss"] - base["rss_before"]
        fd = fsdp0["peak_rss"] - fsdp0["rss_before"]
        print("-" * len(header))
        print(f"peak-overhead ratio baseline/fsdp: {bd / max(fd, 1):.2f}x")
        print(
            "note: 'peak RSS' includes ~torch/interpreter overhead (~rss_before); "
            "'peak-overhead' isolates model+training; 'activations~' = peak-overhead - resident."
        )

    _report_categories(base, fsdp0)
    _report_fsdp_internal(fsdp0)


def _report_categories(base: dict | None, fsdp0: dict | None) -> None:
    """Granular per-category peak tensor memory (from the profiler's memory
    timeline): where the bytes actually live — params, optimizer state, grads,
    activations, autograd-saved tensors, temporaries."""
    if base is None and fsdp0 is None:
        return
    print("\n" + "#" * 78)
    print("# GRANULAR PEAK TENSOR MEMORY  (per category, from profiler memory timeline)")
    print("#" * 78)
    header = f"{'category':<18}{'baseline':>16}{'fsdp/rank':>16}{'base/fsdp':>12}"
    print(header)
    print("-" * len(header))
    for name, label in CATEGORY_ORDER + [("TOTAL", "TOTAL (co-peak)")]:
        key = f"cat_{name}"
        b = base.get(key, 0) if base else 0
        f = fsdp0.get(key, 0) if fsdp0 else 0
        ratio = f"{b / f:.2f}x" if f else "-"
        print(f"{label:<18}{_fmt(b):>16}{_fmt(f):>16}{ratio:>12}")
    print("-" * len(header))
    print(
        "note: each category peak is its own high-water mark; TOTAL is the max of\n"
        "      the SUM of all live tensors at one instant (categories don't co-peak).\n"
        "      FSDP shards PARAMETER/OPTIMIZER_STATE (~1/world_size) but GRADIENT\n"
        "      stays high — the async reduce-scatter keeps full grads live until\n"
        "      finish_gradient_synchronization."
    )


def _report_fsdp_internal(fsdp0: dict | None) -> None:
    """Print FSDP-specific transient buffers sampled during the first step."""
    if not fsdp0:
        return
    samples = fsdp0.get("fsdp_internal_samples") or []
    if not samples:
        return

    fields = [
        ("local_shard_params", "shard params"),
        ("local_shard_grads", "shard grads"),
        ("forward_all_gather_buffers", "fwd all-gather"),
        ("backward_all_gather_buffers", "bwd all-gather"),
        ("active_full_params", "active full params"),
        ("pending_reduce_scatter_full_grad_inputs", "RS full-grad inputs"),
        ("pending_reduce_scatter_shard_outputs", "RS shard outputs"),
        ("tracked_total", "tracked total"),
    ]

    print("\n" + "#" * 78)
    print("# FSDP INTERNAL TRANSIENT BUFFERS  (rank 0, sampled during first step)")
    print("#" * 78)
    header = f"{'point':<30}" + "".join(f"{label:>16}" for _, label in fields)
    print(header)
    print("-" * len(header))
    for sample in samples:
        row = f"{sample['point']:<30}"
        for key, _ in fields:
            row += f"{_fmt(sample.get(key, 0)):>16}"
        print(row)
    print("-" * len(header))
    print(
        "note: these are FSDP-specific tensors only. The reduce-scatter input is\n"
        "      the full flattened gradient that must stay alive until the async\n"
        "      collective completes; the shard output becomes the local grad."
    )


def main() -> None:
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Separate processes so each phase gets a clean RSS high-water mark.
    mp.spawn(run_phase, args=("baseline", 1), nprocs=1, join=True)
    mp.spawn(run_phase, args=("fsdp", WORLD_SIZE), nprocs=WORLD_SIZE, join=True)

    _report()


if __name__ == "__main__":
    main()
