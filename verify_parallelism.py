#!/usr/bin/env python
"""
Verify tensor and pipeline parallelism by comparing with no-parallelism baselines.

Pipeline baseline: run all layers sequentially on the full batch in one process.
Tensor baseline: construct the effective full weights ([A|A|...|A] due to seed=0
init giving every rank the same shard) and run in one process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import re
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from parallelism import generate_sample_data, get_init_params

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def pipeline_baseline(num_layers, num_steps):
    """
    Single-process baseline for pipeline parallelism.

    Pipeline parallelism distributes layers across ranks and splits the batch
    into micro-batches. With gradient accumulation, the result is mathematically
    equivalent to processing the full batch through all layers at once.
    """
    torch.manual_seed(42)  # Must match the monkey-patched seed in _verify_wrapper
    data = generate_sample_data()
    num_dim = data.shape[1]
    params = [get_init_params(num_dim, num_dim, 0) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=0.001)

    results = []
    for step in range(num_steps):
        optimizer.zero_grad()
        x = data
        for layer in range(num_layers):
            x = x @ params[layer]
            x = F.relu(x)
        loss = x.square().sum()
        loss.backward()
        optimizer.step()
        results.append(params[0][0][:3].detach().clone())
    return results


def tensor_baseline(num_layers, num_steps, world_size):
    """
    Single-process baseline for tensor parallelism.

    Each rank holds a column shard [num_dim, local_dim]. Since get_init_params
    uses seed=0 and ignores rank, every rank has the *same* shard A.
    The effective full weight per layer is [A | A | ... | A].
    """
    torch.manual_seed(42)  # Must match the monkey-patched seed in _verify_wrapper
    data = generate_sample_data()
    num_dim = data.shape[1]
    local_dim = num_dim // world_size

    params = []
    for _ in range(num_layers):
        shard = get_init_params(num_dim, local_dim, 0)
        full = nn.Parameter(torch.cat([shard.data] * world_size, dim=1))
        params.append(full)

    optimizer = torch.optim.AdamW(params, lr=0.001)

    results = []
    for step in range(num_steps):
        optimizer.zero_grad()
        x = data
        for layer in range(num_layers):
            x = x @ params[layer]
            x = F.relu(x)
        loss = x.square().sum()
        loss.backward()
        optimizer.step()
        # First 3 elements of full weight's first row = first 3 of shard's first row
        results.append(params[0][0][:3].detach().clone())
    return results

# ---------------------------------------------------------------------------
# Run distributed versions via subprocess
# ---------------------------------------------------------------------------

def parse_params(output, rank=0):
    """Extract param tensors from parallel run stdout."""
    results = {}
    pattern = rf'\[Step (\d+)\] Rank {rank} has params tensor\(\[([^\]]+)\]'
    for line in output.split('\n'):
        m = re.search(pattern, line)
        if m:
            step = int(m.group(1))
            vals = [float(v.strip()) for v in m.group(2).split(',')]
            results[step] = torch.tensor(vals)
    return results


def run_parallel(python_exe, name, world_size, num_layers, num_steps, num_micro_batches=None):
    # Use _verify_wrapper to set torch print precision in spawned child processes.
    if name == 'pipeline':
        code = f"""\
import torch.multiprocessing as mp, sys
sys.path.insert(0, {SCRIPT_DIR!r})
from parallelism import generate_sample_data
from _verify_wrapper import pipeline_wrapper
data = generate_sample_data()
mp.spawn(pipeline_wrapper,
         args=({world_size}, data, {num_layers}, {num_steps}, {num_micro_batches}),
         nprocs={world_size}, join=True)
"""
    else:
        code = f"""\
import torch.multiprocessing as mp, sys
sys.path.insert(0, {SCRIPT_DIR!r})
from parallelism import generate_sample_data
from _verify_wrapper import tensor_wrapper
data = generate_sample_data()
mp.spawn(tensor_wrapper,
         args=({world_size}, data, {num_layers}, {num_steps}),
         nprocs={world_size}, join=True)
"""
    result = subprocess.run(
        [python_exe, '-c', code],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=120,
    )
    if result.returncode != 0:
        print(f"  ERROR (exit code {result.returncode}):")
        err = result.stderr
        print(err[-2000:] if len(err) > 2000 else err)
        return {}, result.stdout
    return parse_params(result.stdout), result.stdout

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(baseline, parallel, num_steps, atol=1e-4):
    ok = True
    for step in range(num_steps):
        b = baseline[step]
        if step not in parallel:
            print(f"  Step {step}: MISSING from parallel output")
            ok = False
            continue
        p = parallel[step]
        match = torch.allclose(b, p, atol=atol, rtol=1e-4)
        tag = "PASS" if match else "FAIL"
        print(f"  Step {step}: {tag}")
        print(f"    baseline = {b.tolist()}")
        print(f"    parallel = {p.tolist()}")
        if not match:
            print(f"    diff     = {(b - p).abs().tolist()}")
            ok = False
    return ok

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    python_exe = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    world_size = 2
    num_layers = 4
    num_steps = 2
    num_micro_batches = 4

    print("=" * 60)
    print("PIPELINE PARALLELISM")
    print("=" * 60)
    print("Computing baseline...")
    pb = pipeline_baseline(num_layers, num_steps)
    print("Running distributed version...")
    pp, stdout = run_parallel(python_exe, 'pipeline', world_size, num_layers, num_steps, num_micro_batches)
    print("Comparison (params[0][0][:3] at rank 0):")
    pipe_ok = compare(pb, pp, num_steps)

    print()
    print("=" * 60)
    print("TENSOR PARALLELISM")
    print("=" * 60)
    print("Computing baseline...")
    tb = tensor_baseline(num_layers, num_steps, world_size)
    print("Running distributed version...")
    tp, stdout = run_parallel(python_exe, 'tensor', world_size, num_layers, num_steps)
    print("Comparison (params[0][0][:3] at rank 0):")
    tens_ok = compare(tb, tp, num_steps)

    print()
    print("=" * 60)
    print(f"Pipeline: {'PASS' if pipe_ok else 'FAIL'}")
    print(f"Tensor:   {'PASS' if tens_ok else 'FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
