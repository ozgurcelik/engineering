# %%
import os
import re
import json
import hashlib
import shutil
import traceback
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, List, Any, Union
import torch


def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x, 1)


def mean(x: List[float]) -> float:
    return sum(x) / len(x)


def count(list, x):
    """Return the number of times `x` appears in `list`."""
    return sum(1 for y in list if y == x)


def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def download_file(url: str, filename: str):
    """Download `url` and save the contents to `filename`.  Skip if `filename` already exists."""
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        response = requests.get(url)
        with open(filename, "wb") as f:
            shutil.copyfileobj(BytesIO(response.content), f)


def cached(url: str) -> str:
    """Download `url` if needed and return the location of the cached file."""
    name = re.sub(r"[^\w_-]+", "_", url)
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()

    path = os.path.join("var", url_hash + "-" + name)
    download_file(url, path)
    return path


def get_stack(pop_stack: bool = False):
    """
    Return the current stack as a string.
    if `pop_stack`, then remove the last function.
    """
    stack = traceback.extract_stack()
    # Start at <module>
    i = None
    for j, frame in enumerate(stack):
        if frame.name == "<module>":
            i = j
    if i is not None:
        stack = stack[i + 1:]  # Delete everything up to the last module
        stack = stack[:-2]  # Remove the current two functions (get_stack and point/figure/etc.)
    if pop_stack:
        stack = stack[:-1]
    stack = [
        {
            "name": frame.name,
            "filename": os.path.basename(frame.filename),
            "lineno": frame.lineno,
        } \
        for frame in stack
    ]
    return stack


def note(message: str, style: Optional[dict] = None, verbatim: bool = False, pop_stack: bool = False):
    """Make a note (bullet point) with `message`."""
    print("note:", message)

    style = style or {}
    if verbatim:
        messages = message.split("\n")
        style = {
            "font-family": "monospace",
            "white-space": "pre",
            **style
        }
    else:
        messages = [message]

    for message in messages:
        stack = get_stack(pop_stack=pop_stack)
        add_content("addText", [stack, message, style])


def see(obj: Any, pop_stack: bool = False):
    """References `obj` in the code, but don't print anything out."""
    print("see:", obj)

    if isinstance(obj, str):
        message = obj
    else:
        message = str(obj)
    style = {"color": "gray"}

    stack = get_stack(pop_stack=pop_stack)
    add_content("addText", [stack, message, style])


def image(path: str, style: Optional[dict] = None, width: float = 1.0, pop_stack: bool = False):
    """Show the image at `path`."""
    print("image:", path)

    style = style or {}
    style["width"] = str(width * 100) + "%"

    stack = get_stack(pop_stack=pop_stack)
    add_content("addImage", [stack, path, style])


# Where the contents of the lecture are written to be displayed via `view.html`.
content_path: Optional[str] = None

def init_content(path: str):
    global content_path
    content_path = path
    # Clear the file
    with open(content_path, "w") as f:
        pass

def add_content(function_name, args: List[Any]):
    assert content_path
    line = function_name + "(" + ", ".join(map(json.dumps, args)) + ")"
    # Append to the file
    with open(content_path, "a") as f:
        print(line, file=f)

############################################################

@dataclass(frozen=True)
class Spec:
    name: Optional[str] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    description: Optional[Union[str, List[str]]] = None
    references: Optional[List[Any]] = None


@dataclass(frozen=True)
class MethodSpec(Spec):
    pass


@dataclass(frozen=True)
class DataSpec(Spec):
    num_tokens: Optional[int] = None
    vocabulary_size: Optional[int] = None


@dataclass(frozen=True)
class ArchitectureSpec(Spec):
    num_parameters: Optional[int] = None
    num_layers: Optional[int] = None
    dim_model: Optional[int] = None
    num_heads: Optional[int] = None
    dim_head: Optional[int] = None
    description: Optional[str] = None
    references: Optional[List[Any]] = None


@dataclass(frozen=True)
class TrainingSpec(Spec):
    context_length: Optional[int] = None
    batch_size_tokens: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None
    hardware: Optional[str] = None
    num_epochs: Optional[int] = None
    num_flops: Optional[int] = None
    references: Optional[List[Any]] = None


@dataclass(frozen=True)
class ModelSpec(Spec):
    data: Optional[DataSpec] = None
    architecture: Optional[ArchitectureSpec] = None
    training: Optional[TrainingSpec] = None
# %%
import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import os
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
    note("While benchmarking looks at end-to-end time, profiling looks at where time is spent.")
    note("Obvious: profiling helps you understand where time is being spent.")
    note("Deeper: profiling helps you understand (what is being called).")

    note("PyTorch has a nice built-in profiler"), see("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")

    note("Let's profile some code to see what is going on under the hood.")
    profile("sleep", lambda : time.sleep(50 / 1000))  # Dummy function

    note("Let's start with some basic operations.")
    profile("add", run_operation2(dim=2048, operation=lambda a, b: a + b))
    profile("matmul", run_operation2(dim=2048, operation=lambda a, b: a @ b))
    profile("matmul(dim=128)", run_operation2(dim=128, operation=lambda a, b: a @ b))

    note("Observations")
    note("- You can see what CUDA kernels are actually being called.")
    note("- Different CUDA kernels are invoked depending on the tensor dimensions.")

    note("Name of CUDA kernel tells us something about the implementation.")
    note("Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1")
    note("- cutlass: NVIDIA's CUDA library for linear algebra")
    note("- 256x128: tile size")

    note("Let's now look at some composite operations.")
    profile("cdist", run_operation2(dim=2048, operation=lambda a, b: torch.cdist(a, b)))
    profile("gelu", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.gelu(a + b)))
    profile("softmax", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.softmax(a + b)))

    note("Now let's profile our MLP.")
    note("We will also visualize our stack trace using a flame graph, which reveals where time is being spent.")
    if torch.cuda.is_available():
        profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
    else:
        profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)
# %%
ensure_directory_exists("var")
init_content("var/gpus.js")
profiling()
# %%
