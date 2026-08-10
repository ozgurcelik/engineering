"""Render static blog figures from the profiler's interactive HTML timelines."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fsdp-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fsdp-cache")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "figures"

PLOTS = [
    (
        ROOT / "results" / "mlp" / "baseline_rank0_memory.html",
        OUTPUT_DIR / "mlp_baseline_memory_timeline.png",
        "Deep MLP — baseline (rank 0)",
    ),
    (
        ROOT / "results" / "mlp" / "fsdp_rank0_memory.html",
        OUTPUT_DIR / "mlp_fsdp_memory_timeline.png",
        "Deep MLP — FSDP (rank 0)",
    ),
    (
        ROOT / "results" / "transformer" / "baseline_seq1024_rank0_memory.html",
        OUTPUT_DIR / "transformer_seq1024_baseline_memory_timeline.png",
        "Transformer, sequence length 1024 — baseline (rank 0)",
    ),
    (
        ROOT / "results" / "transformer" / "fsdp_seq1024_rank0_memory.html",
        OUTPUT_DIR / "transformer_seq1024_fsdp_memory_timeline.png",
        "Transformer, sequence length 1024 — FSDP (rank 0)",
    ),
]


def load_timeline(path: Path) -> dict[str, object]:
    html = path.read_text()
    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n    document.getElementById", start)
    return json.loads(html[start:end])


def render(source: Path, destination: Path, title: str) -> None:
    data = load_timeline(source)
    points = data["points"]
    categories = data["categories"]
    labels = [
        "autograd internals" if label == "autograd saved" else label
        for label in data["labels"]
    ]
    colors = data["colors"]

    times = [point[0] for point in points]
    series = [
        [point[index + 1] for point in points]
        for index in range(len(categories))
    ]

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=180)
    ax.stackplot(times, *series, labels=labels, colors=colors, step="post", alpha=0.94)
    ax.set_title(title, loc="left", fontsize=16, fontweight="semibold", pad=14)
    ax.set_xlabel("Profile time (ms)")
    ax.set_ylabel("Live tensor memory (MiB)")
    ax.set_xlim(min(times), max(times))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.4,
        columnspacing=1.4,
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for source, destination, title in PLOTS:
        render(source, destination, title)
