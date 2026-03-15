# %%
import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import os
from util import *
# %%
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU available")
# %%

class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())

    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            y.backward()

    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x, y)
# %%
def create_flame_graph(in_path: str, out_path: str):
    """Create a flame graph from the profiler output in `in_path` and output a SVG file to `out_path`."""
    # https://www.brendangregg.com/flamegraphs.html
    if not os.path.exists("FlameGraph"):
        os.system("git clone https://github.com/brendangregg/FlameGraph")
    os.system(f"FlameGraph/flamegraph.pl --title \"CUDA time\" --countname \"us\" {in_path} > {out_path}")

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    note(f"## {description}", pop_stack=True)
    note(table, verbatim=True, pop_stack=True)

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
        create_flame_graph(text_path, svg_path)
        image(svg_path, width=1, pop_stack=True)
# %%
def profiling():
    # note("While benchmarking looks at end-to-end time, profiling looks at where time is spent.")
    # note("Obvious: profiling helps you understand where time is being spent.")
    # note("Deeper: profiling helps you understand (what is being called).")

    # note("PyTorch has a nice built-in profiler"), see("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")

    # note("Let's profile some code to see what is going on under the hood.")
    # profile("sleep", lambda : time.sleep(50 / 1000))  # Dummy function

    # note("Let's start with some basic operations.")
    # profile("add", run_operation2(dim=2048, operation=lambda a, b: a + b))
    # profile("matmul", run_operation2(dim=2048, operation=lambda a, b: a @ b))
    # profile("matmul(dim=128)", run_operation2(dim=128, operation=lambda a, b: a @ b))

    # note("Observations")
    # note("- You can see what CUDA kernels are actually being called.")
    # note("- Different CUDA kernels are invoked depending on the tensor dimensions.")

    # note("Name of CUDA kernel tells us something about the implementation.")
    # note("Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1")
    # note("- cutlass: NVIDIA's CUDA library for linear algebra")
    # note("- 256x128: tile size")

    # note("Let's now look at some composite operations.")
    # profile("cdist", run_operation2(dim=2048, operation=lambda a, b: torch.cdist(a, b)))
    # profile("gelu", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.gelu(a + b)))
    # profile("softmax", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.softmax(a + b)))

    # note("Now let's profile our MLP.")
    # note("We will also visualize our stack trace using a flame graph, which reveals where time is being spent.")
    if torch.cuda.is_available():
        note("Profiling MLP with dimensions 2048x2048 and 64 layers, batch size 1024, and 2 steps.")
        profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
    else:
        note("Profiling MLP with dimensions 128x128 and 16 layers, batch size 128, and 2 steps.")
        profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)
# %%
ensure_directory_exists("var")
init_content("var/gpus.js")
profiling()
# %%
