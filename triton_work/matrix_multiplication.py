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
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_SIZE_N
    a_start_ptr = a_ptr + row * a_row_stride
    b_start_ptr = b_ptr

    n_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    n_mask = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets * a_col_stride # 1d array with length BLOCK_SIZE_K
        b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride # 2d array with shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_mask = k_mask
        b_mask = k_mask[:, None] & n_mask[None, :]

        a_vals = tl.load(a_ptrs, mask=a_mask) # shape [BLOCK_SIZE_K]
        b_vals = tl.load(b_ptrs, mask=b_mask) # shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        acc += tl.sum(a_vals[:, None] * b_vals, axis=0) # shape [BLOCK_SIZE_N]

    c_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    c_mask = c_offsets < N
    c_m_n_ptr = c_ptr + row * c_row_stride + c_offsets * c_col_stride
    tl.store(c_m_n_ptr, acc, mask=c_mask)

def matrix_multiplication_naive_row_major(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    grid_size = (M, triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty(M, N, device=DEVICE)
    matrix_multiplication_kernel_naive_row_major[grid_size](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), a.stride(1), b.stride(1), c.stride(1), BLOCK_SIZE_N, BLOCK_SIZE_K)
    return c


@triton.jit
def matrix_multiplication_tiled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_SIZE_M
    col = tl.program_id(1) * BLOCK_SIZE_N

    m_offsets = tl.arange(0, BLOCK_SIZE_M) + row
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    n_mask = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = m_offsets[:, None] * a_row_stride + k_offsets[None, :] * a_col_stride # shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride # shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        a_ptrs = a_ptr + a_offsets
        b_ptrs = b_ptr + b_offsets

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]
        
        a_vals = tl.load(a_ptrs, mask=a_mask)
        b_vals = tl.load(b_ptrs, mask=b_mask)

        acc = tl.dot(a_vals, b_vals, acc, input_precision="ieee")

    c_offsets = m_offsets[:, None] * c_row_stride + n_offsets[None, :] * c_col_stride
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)

def matrix_multiplication_tiled(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty(M, N, device=DEVICE)
    matrix_multiplication_tiled_kernel[grid_size](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), a.stride(1), b.stride(1), c.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    return c


# %%
torch.manual_seed(0)
a = torch.randn((512, 192), device=DEVICE, dtype=torch.float32)
b = torch.randn((192, 128), device=DEVICE, dtype=torch.float32)
c_triton = matrix_multiplication_naive(a, b)
c_triton_blocked = matrix_multiplication_naive_blocked(a, b)
c_triton_row_major = matrix_multiplication_naive_row_major(a, b)
c_triton_tiled = matrix_multiplication_tiled(a, b)
c_torch = torch.matmul(a, b)
assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2), (c_triton, c_torch)
assert torch.allclose(c_triton_blocked, c_torch, atol=1e-2, rtol=1e-2), (c_triton_blocked, c_torch)
assert torch.allclose(c_triton_row_major, c_torch, atol=1e-2, rtol=1e-2), (c_triton_row_major, c_torch)
assert torch.allclose(c_triton_tiled, c_torch, atol=1e-1, rtol=1e-1), (c_triton_tiled, c_torch)
# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 18)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'triton_blocked', 'triton_row_major', 'triton_tiled', 'torch'],  # possible values for `line_arg`
        line_names=["Triton", "Triton Blocked", "Triton Row Major", "Triton Tiled", "Torch"],  # label name for the lines
        styles=[('blue', '-'), ('red', '-'), ('purple', '-'), ('orange', '-'), ('green', '-')],  # line styles
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
    if provider == 'triton_row_major':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_naive_row_major(a, b))
    if provider == 'triton_tiled':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_tiled(a, b))
    # FLOPs for matmul: 2 * M * N * K (one multiply + one add per output element per K dim)
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark.run(show_plots=True, print_data=True)
# %%