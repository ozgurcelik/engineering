#%%
import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# %%
# calculate each element of the result matrix separately by loading each element of A and B once
@triton.jit
def matrix_multiplication_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = 0.0
    for k in range(K):
        a_val = tl.load(a_ptr + row * a_row_stride + k)
        b_val = tl.load(b_ptr + k * b_row_stride + col)
        acc += a_val * b_val
    c_m_n_ptr = c_ptr + row * c_row_stride + col
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE)
    matrix_multiplication_kernel_naive[(M, N)](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0))
    return c


# calculate the each element of the result matrix separately by loading blocks of A and B
@triton.jit
def matrix_multiplication_kernel_naive_blocked(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = 0.0
    for k in range(0, K, BLOCK_SIZE_K):
        a_start_ptr = a_ptr + row * a_row_stride
        b_start_ptr = b_ptr + col

        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets
        b_offsets = k_offsets * b_row_stride

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_vals = tl.load(a_ptrs, mask=k_mask)
        b_vals = tl.load(b_ptrs, mask=k_mask)

        acc += tl.sum(a_vals * b_vals)

    c_m_n_ptr = c_ptr + row * c_row_stride + col
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive_blocked(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_K = 128
    c = torch.empty(M, N, device=DEVICE)
    matrix_multiplication_kernel_naive_blocked[(M, N)](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), BLOCK_SIZE_K)
    return c


# do it row major. load a row of A once in a kernel and calculate the entire row of the result matrix in one kernel
@triton.jit
def matrix_multiplication_kernel_naive_row_major(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0)
    a_start_ptr = a_ptr + row * a_row_stride
    b_start_ptr = b_ptr

    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets
        b_offsets = k_offsets * b_row_stride + tl.arange(0, BLOCK_SIZE_N)

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_vals = tl.load(a_ptrs, mask=k_mask)
        b_vals = tl.load(b_ptrs, mask=k_mask)

        acc += a_vals * b_vals
    c_m_n_ptr = c_ptr + row * c_row_stride + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive_row_major(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_N = N
    BLOCK_SIZE_K = 128
    c = torch.empty(M, N, device=DEVICE)
    matrix_multiplication_kernel_naive_row_major[(M, N)](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), BLOCK_SIZE_N, BLOCK_SIZE_K)
    return c
# %%
torch.manual_seed(0)
a = torch.randn((512, 192), device=DEVICE, dtype=torch.float32)
b = torch.randn((192, 128), device=DEVICE, dtype=torch.float32)
c_triton = matrix_multiplication_naive(a, b)
c_triton_blocked = matrix_multiplication_naive_blocked(a, b)
c_triton_row_major = matrix_multiplication_naive_row_major(a, b)
c_torch = torch.matmul(a, b)
assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2), (c_triton, c_torch)
assert torch.allclose(c_triton_blocked, c_torch, atol=1e-2, rtol=1e-2), (c_triton_blocked, c_torch)
assert torch.allclose(c_triton_row_major, c_torch, atol=1e-2, rtol=1e-2), (c_triton_row_major, c_torch)
# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 17)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'triton_blocked', 'torch'],  # possible values for `line_arg`
        line_names=["Triton", "Triton Blocked", "Torch"],  # label name for the lines
        styles=[('blue', '-'), ('red', '-'), ('green', '-')],  # line styles
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_naive(a, b))
    if provider == 'triton_blocked':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_naive_blocked(a, b))
    # FLOPs for matmul: 2 * M * N * K (one multiply + one add per output element per K dim)
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark.run(show_plots=True, print_data=True)
# %%