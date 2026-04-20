#%%
import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# %%
def naive_softmax(x: torch.Tensor):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read MN elements, write M
    xmax = x.max(dim=1)[0] # if x is shape MN, xmax is shape M
    # read MN + M elements, write MN
    z = x - xmax[:, None]
    # read MN, write MN
    numerator = torch.exp(z)
    # read MN, write M
    denominator = numerator.sum(dim=1)
    # read MN + M, write MN
    result = numerator / denominator[:, None]
    # in total, read 5MN + 2M, write 3MN + 2M
# %%
x = torch.randn(1823, 781)
output = naive_softmax(x)
# %%
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
# %%
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    # each program uses n_regs * (WARP_SIZE * num_warps) registers total (registers per thread × threads per program). 
    # NUM_REGS is the SM's register file size. Dividing gives how many programs fit per SM based on registers.
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    # if each program needs size_smem bytes of shared memory and the SM has SIZE_SMEM bytes total, only SIZE_SMEM // size_smem fit 
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y
# %%
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)
# %%
## Comments

# Q1: Where do we set how many warps per program?
# --------------------------------------------------
# Via the `num_warps` kwarg on the kernel launch (see line 77 and line 87).
# It's a compile-time launch attribute, baked into the PTX during `.warmup(...)`.
# Triton's equivalent of CUDA's blockDim, but expressed in warps instead of threads:
#   num_warps=8  ->  8 warps * 32 threads = 256 threads per program.
# Changing it requires a recompile; that's why it's passed alongside BLOCK_SIZE
# and num_stages.

# Q2: How does `num_warps` affect the kernel body (lines 37-56)?
# --------------------------------------------------------------
# It doesn't appear in the source at all -- that's the point of Triton's tile
# abstraction. `col_offsets = tl.arange(0, BLOCK_SIZE)` and `row = tl.load(...)`
# describe a BLOCK_SIZE-wide tile that belongs to the whole program. The compiler
# then distributes tile-wide ops across `num_warps * 32` threads however it likes.
# Block-wide reductions like `tl.max` / `tl.sum` are lowered to a tree: each warp
# first reduces its own 32 lanes via warp shuffles, then the per-warp partials
# are combined through shared memory to produce the final scalar.
#
# Example with BLOCK_SIZE=1024:
#   num_warps=8  (256 threads):  each thread handles 4 tile elements; reduction
#                                tree has 8 warps collaborating via warp shuffles.
#   num_warps=4  (128 threads):  each thread handles 8 tile elements; 4-warp
#                                reduction, more registers per thread.
#
# So `num_warps` changes codegen and performance, never correctness:
#   - more warps  -> more parallelism per block, but fewer registers/thread and
#                    fewer blocks fit per SM (see line 93 occupancy formula).
#   - fewer warps -> more register budget per thread, less parallelism in the
#                    reduction, potentially slower loads/stores.

# Q3: What if BLOCK_SIZE < n_cols?
# --------------------------------
# The kernel silently produces WRONG output. It only ever loads the first
# BLOCK_SIZE elements of each row:
#   - tl.max / tl.sum compute over a truncated row -> wrong normalizer.
#   - tl.store only writes the first BLOCK_SIZE output columns; the rest stay
#     as uninitialized garbage from `torch.empty_like(x)`.
# The mask only protects the BLOCK_SIZE > n_cols case (overshoot). There's no
# outer loop to advance past BLOCK_SIZE columns, so undershoot is unprotected.
#
# That's why the launcher picks BLOCK_SIZE = next_power_of_2(n_cols) at line 71
# -- it guarantees BLOCK_SIZE >= n_cols always holds.
#
# Implicit limit: the full row must fit in registers. Fine for hidden dims up to
# a few thousand (BERT, small LLMs), breaks down for 100K+ vocab softmax. The
# industrial fix is "online softmax" (streaming chunks with running max + sum,
# the trick behind FlashAttention), which is outside this tutorial's scope.

# Q4: How is `n_regs` (line 89) calculated, and how does it relate to BLOCK_SIZE?
# --------------------------------------------------------------------------------
# `n_regs` is the per-thread register count chosen by `ptxas` during register
# allocation. `.warmup(...)` runs the full compile pipeline
#   Triton IR -> TritonGPU IR -> LLVM IR -> PTX -> SASS (ptxas allocates regs)
# without launching, and the final count is exposed as `kernel.n_regs`. It's the
# same number you'd see from `ptxas -v` or `cuobjdump --dump-resource-usage`.
#
# For a tile-heavy kernel like this one it scales roughly as:
#   n_regs ~ c0 + c1 * num_stages * (BLOCK_SIZE / (num_warps * 32))
# i.e. linear in elements-per-thread. c0 (~20-30) is scalar overhead (pointers,
# loop counter, reduced max/sum); c1 (~2-4) is how many live tiles the compiler
# can't fuse away (`row`, `row_minus_max`, `numerator`, etc).

# Q5: Register spilling -- what is it and why does it kill performance?
# ---------------------------------------------------------------------
# Every thread has a hard cap of 255 registers on modern NVIDIA GPUs. If the
# compiler decides a thread needs more live values than that, the overflow
# "spills" to local memory -- which is a per-thread slice of GLOBAL DRAM,
# cached in L1/L2. Rough latencies: register ~1 cycle, L1 ~30, L2 ~200,
# DRAM ~500-800. A spilled access is 30-800x slower than a register.
#
# Even without spilling, high n_regs crushes occupancy via the formula at
# line 93: on an H100 with 65536 regs/SM and num_warps=8 (256 threads/program),
#   n_regs=32  -> 8 programs/SM
#   n_regs=64  -> 4 programs/SM
#   n_regs=128 -> 2 programs/SM
#   n_regs=255 -> 1 program/SM
# Fewer resident programs means less latency hiding, which matters a lot for
# memory-bound kernels like softmax.
#
# Concrete spill scenario: BLOCK_SIZE=16384, num_warps=4 (128 threads) gives
# 128 elts/thread. With ~3 live tiles that's ~384 tile regs alone -> well over
# 255 -> ptxas spills. Fix: bump num_warps to 16 or 32, so elts/thread drops
# to 16 or 8 and n_regs stays comfortably under the cap. This is exactly the
# tradeoff space the autotuner sweeps in the next tutorial.