# %%
import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import os
from util import *
import triton
import triton.language as tl
# %%
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU available")

max_threads_per_block = torch.cuda.get_device_properties(0).max_threads_per_block  # 1024
print(f"## max_threads_per_block {max_threads_per_block}")
# %%

def check_equal(f1, f2):
    x = torch.randn(2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times

    mean_time = mean(times) # @inspect mean_time
    return mean_time


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
def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

def create_cuda_gelu():
    # Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Read the CUDA kernel source code
    cuda_gelu_src = open("cuda_kernels/gelu.cu").read()

    # C++ code: defines the gelu function
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    # Compile the CUDA code and bind it to a Python module.
    ensure_directory_exists("var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )

    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu

def kernel_fusion_motivation():

    print("Let's consider two ways to compute GeLU:")
    x = torch.tensor([1.])  # @inspect x

    print("1. The default PyTorch implementation (fused):")
    y1 = pytorch_gelu(x)  # @inspect y1

    print("2. We can also write our own by hand (not fused):")
    y2 = manual_gelu(x)  # @inspect y2

    # Check that the implementations match
    assert torch.allclose(y1, y2)

    # Check more systematically
    check_equal(pytorch_gelu, manual_gelu)

    print("Let's benchmark.")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    if manual_time is not None and pytorch_time is not None:
        print(f"The fused version is significantly faster: {manual_time:.2f} ms, {pytorch_time:.2f} ms")
    else:
        print("Could not compare times - benchmark results were None")

    print("Let's look under the hood.")
    manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    print(f"## manual_gelu")
    print(manual_gelu_profile)
    pytorch_gelu_profile = profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    print(f"## pytorch_gelu")
    print(pytorch_gelu_profile)
    print("The PyTorch just calls one kernel whereas the others are atomic (remember the warehouse/factory) ")

    print(f"## Look at Nsight profiler for MLP   ")

def cuda_kernels():
    print("Now let's open the box to understand what's going on inside a CUDA kernel by writing our own.")

    print("Let's write the GeLU function in CUDA.")
    cuda_gelu = create_cuda_gelu() # @inspect cuda_gelu
    x = manual_gelu # @inspect x

    print("Check correctness of our implementation.")
    if cuda_gelu is not None:
        check_equal(cuda_gelu, manual_gelu)

    print("Benchmark our CUDA version.")
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    print(f"## pytorch_gelu time {pytorch_time}")
    print(f"## manual_gelu time {manual_time}")
    if cuda_gelu is not None:
        cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu)) # @inspect cuda_time 
        cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))
        print(f"## cuda_gelu")
        print(cuda_gelu_profile)
    print("Our CUDA implementation is faster than manual, but not as good as PyTorch.")

    print("Elementwise operations are easy in CUDA (though you can still be smarter).")
    print("But most interesting operations (e.g., matmul, softmax, RMSNorm) require reading multiple values.")
    print("For that, you have to think about managing shared memory, etc.")

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Handle boundary
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu_main():
    if not torch.cuda.is_available():
        return

    print("One big advantage of Triton is that you can step through the Python code.")

    print("Let's step through a Triton kernel.")
    x = torch.randn(8192, device=get_device())
    y1 = triton_gelu(x)

    print("Check that it's correct.")
    check_equal(triton_gelu, manual_gelu)

    print("Let's now benchmark it compared to the PyTorch and CUDA implementations.")
    print("Remember to set TRITON_INTERPRET=0 for good performance.")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time
    print(f"## manual_gelu time {manual_time}")
    print(f"## pytorch_gelu time {pytorch_time}")
    print(f"## cuda_gelu time {cuda_time}")
    print(f"## triton_gelu time {triton_time}")

    triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    print(f"## triton_gelu")
    print(triton_gelu_profile)

    print("Our Triton implementation (triton_gelu):")
    print("- is almost as good as the PyTorch implementation (pytorch_gelu).")
    print("- is actually slower than our naive CUDA implementation (cuda_gelu).")

    print("Triton operates on blocks, CUDA operates on threads.")
    print("Blocks allows Triton compiler to do other optimizations (e.g., thread coarsening).")

    print("Everything is way faster than the manual implementation (manual_gelu).")

def pytorch_compilation():

    print("- Write it in Python and compile it into Triton")
    compiled_gelu = torch.compile(manual_gelu)

    print("Check correctness of our implementation.")
    check_equal(compiled_gelu, manual_gelu)

    if not torch.cuda.is_available():
        return

    print("Let's benchmark and profile it!")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time
    compiled_time = benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time

    print(f"## manual_gelu time {manual_time}")
    print(f"## pytorch_gelu time {pytorch_time}")
    print(f"## cuda_gelu time {cuda_time}")
    print(f"## triton_gelu time {triton_time}")
    print(f"## compiled_gelu time {compiled_time}")

    print("Let's look under the hood")
    compiled_gelu_profile = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
    print(f"## compiled_gelu")
    print(compiled_gelu_profile)
# %%
ensure_directory_exists("var")
init_content("var/gpus.js")
# kernel_fusion_motivation()
# cuda_kernels()
# triton_gelu_main()
pytorch_compilation()