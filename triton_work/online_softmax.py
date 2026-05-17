# %%
# pyright: reportUnreachable=false
import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Imagine x vector. Then softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j = 1 to N.
But, this can be unstablized due to overflow. So, we write it as:

softmax(x_i) = exp(x_i - max_x) / sum(exp(x_j - max_x)) for j = 1 to N.

where max_x is the maximum element in the vector.

This is stable because the exp(x_j - max_x) terms are all less than 1, so the sum of the exp(x_j - max_x) terms is less than N.

Now, say that m_i is the maximum element in the vector from start to i.
And d_i is the sum of the exp(x_j - max_i) for j = 1 to i.
So, d_N = sum(exp(x_j - max_N)) for j = 1 to N which is the denominator of the softmax function.

d_i = \sum_{j=1}^{i} exp(x_j - max_i)
= \sum_{j=1}^{i-1} exp(x_j - max_i) + exp(x_i - max_i)
= \sum_{j=1}^{i-1} exp(x_j - max_i-1) * exp(max_i-1 - max_i) + exp(x_i - max_i)
= d_{i-1} * exp(max_i-1 - max_i) + exp(x_i - max_i)

"""

@triton.jit
def online_softmax_fwd_kernel(
    x_ptr, y_ptr, # Input and output pointers. Both input and output are (M, N) matrices.
    x_stride_row, x_stride_col,
    y_stride_row, y_stride_col,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_start = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(M, N),
        strides=(x_stride_row, x_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )

    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(M, N),
        strides=(y_stride_row, y_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )

    d = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    m = tl.full((BLOCK_SIZE_M, 1), -float('inf'), dtype=tl.float32)
    
    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Per-row max of this tile -> shape (BLOCK_SIZE_M, 1). `tl.max` is a
        # reduction; `tl.maximum` is elementwise (no axis arg).
        m_block = tl.max(x_block, axis=1, keep_dims=True)
        # New running max across [tile so far, current tile].
        m_new = tl.maximum(m, m_block)
        # Rescale the running denominator into the new max, then add the
        # contribution from the current tile (reduced to (BLOCK_SIZE_M, 1)).
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x_block - m_new), axis=1, keep_dims=True)
        m = m_new

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_N))
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(M, N),
        strides=(x_stride_row, x_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )

    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        y_block = (tl.exp(x_block - m) / d).to(x_block.dtype)
        tl.store(y_block_ptr, y_block, boundary_check=(0, 1))

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_N))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_SIZE_N))

"""
For the backward pass, assume we have dL/dy_j = gy_j. Then, we have:

dL/dx_i = \sum_{j=1}^{N} dL/dy_j * dy_j/dx_i
with
dy_j/dx_i = y_j(delta_ij - y_i)

so

dL/dx_i = \sum_{j=1}^{N} gy_j * y_j(delta_ij - y_i)
= gy_i * y_i - y_i * \sum_{j=1}^{N} gy_j * y_j
= y_i * (gy_i - \sum_{j=1}^{N} gy_j * y_j)

"""

@triton.jit
def online_softmax_bwd_kernel(
    gy_ptr,
    y_ptr,
    dx_ptr,
    gy_stride_row, gy_stride_col,
    y_stride_row, y_stride_col,
    dx_stride_row, dx_stride_col,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_start = tl.program_id(0)

    gy_block_ptr = tl.make_block_ptr(
        gy_ptr,
        shape=(M, N),
        strides=(gy_stride_row, gy_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(M, N),
        strides=(y_stride_row, y_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )
    dx_block_ptr = tl.make_block_ptr(
        dx_ptr,
        shape=(M, N),
        strides=(dx_stride_row, dx_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )
    
    cum_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        gy_block = tl.load(gy_block_ptr, boundary_check=(0, 1), padding_option="zero")
        y_block = tl.load(y_block_ptr, boundary_check=(0, 1), padding_option="zero")
        cum_sum += tl.sum(gy_block * y_block, axis=1, keep_dims=True)
    
        gy_block_ptr = tl.advance(gy_block_ptr, (0, BLOCK_SIZE_N))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_SIZE_N))
    
    gy_block_ptr = tl.make_block_ptr(
        gy_ptr,
        shape=(M, N),
        strides=(gy_stride_row, gy_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(M, N),
        strides=(y_stride_row, y_stride_col),
        offsets=(pid_start * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0),
    )
    
    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        gy_block = tl.load(gy_block_ptr, boundary_check=(0, 1), padding_option="zero")
        y_block = tl.load(y_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dx_block = (y_block * (gy_block - cum_sum)).to(y_block.dtype)
        tl.store(dx_block_ptr, dx_block, boundary_check=(0, 1))

        gy_block_ptr = tl.advance(gy_block_ptr, (0, BLOCK_SIZE_N))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_SIZE_N))
        dx_block_ptr = tl.advance(dx_block_ptr, (0, BLOCK_SIZE_N))

class OnlineSoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        M, N = x.shape

        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = 16

        y = torch.zeros_like(x)
        online_softmax_fwd_kernel[(triton.cdiv(M, BLOCK_SIZE_M),)](
            x, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            M, N,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )

        ctx.save_for_backward(y)

        return y

    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor:
        y = ctx.saved_tensors[0]
        M, N = y.shape

        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = 16

        dx = torch.zeros_like(y)

        online_softmax_bwd_kernel[(triton.cdiv(M, BLOCK_SIZE_M),)](
            gy, y, dx,
            gy.stride(0), gy.stride(1),
            y.stride(0), y.stride(1),
            dx.stride(0), dx.stride(1),
            M, N,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )

        return dx
    


# %%
if __name__ == "__main__":
    torch.manual_seed(0)

    shapes = [(1, 16), (4, 128), (32, 1024), (128, 4096)]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        # Looser tolerances for fp16 due to reduced precision. For large N
        # (e.g. 4096), accumulated rounding can reach a few ULPs of fp16, so
        # we allow ~5e-3 absolute/relative slack.
        atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (5e-3, 5e-3)
        for shape in shapes:
            x_ref = torch.randn(shape, dtype=dtype, device=DEVICE, requires_grad=True)
            x_mine = x_ref.detach().clone().requires_grad_(True)
            gy = torch.randn(shape, dtype=dtype, device=DEVICE)

            y_ref = torch.softmax(x_ref, dim=1)
            y_mine = OnlineSoftmaxFunc.apply(x_mine)
            torch.testing.assert_close(y_mine, y_ref, atol=atol, rtol=rtol)

            y_ref.backward(gy)
            y_mine.backward(gy)
            torch.testing.assert_close(x_mine.grad, x_ref.grad, atol=atol, rtol=rtol)

            fwd_err = (y_mine - y_ref).abs().max().item()
            bwd_err = (x_mine.grad - x_ref.grad).abs().max().item()
            row_sums = y_mine.sum(dim=1)
            print(
                f"[OK] OnlineSoftmaxFunc dtype={dtype} shape={shape} "
                f"fwd_max_abs_err={fwd_err:.2e} "
                f"bwd_max_abs_err={bwd_err:.2e} "
                f"row_sum_min={row_sums.min().item():.6f} "
                f"row_sum_max={row_sums.max().item():.6f}"
            )

    print("All correctness checks passed.")

# %%
# Benchmark in fp32: online softmax (Triton) vs torch.softmax.
#
# Softmax is memory-bound: each element is read once and written once, with a
# small amount of arithmetic in between (max + exp + sum + div). The relevant
# throughput metric is therefore effective DRAM bandwidth in GB/s, not TFLOPS.
# For an (M, N) input in fp32 the forward moves 2 * M * N * 4 bytes per call
# (read x, write y); the backward moves 3 * M * N * 4 bytes per call (read gy,
# read y, write dx).
#
# We hold M fixed (so each row of work is well-sized for the SM) and sweep N,
# which is both the reduction dimension of softmax and the per-row working set.
# Small N is launch-overhead / occupancy bound; large N is DRAM bound and is
# where the comparison against torch's fused softmax is most informative.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=[
            'Triton (online_softmax)',
            'Torch (torch.softmax)',
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='softmax-fwd-performance-fp32',
        args={'M': 4096, 'dtype': torch.float32},
    ))
def benchmark_fwd(M, N, dtype, provider):
    x = torch.randn((M, N), device=DEVICE, dtype=dtype)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: OnlineSoftmaxFunc.apply(x))
    # Bytes moved: read x once, write y once -> 2 * M * N * sizeof(dtype).
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=[
            'Triton (online_softmax bwd)',
            'Torch (torch.softmax bwd)',
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='softmax-bwd-performance-fp32',
        args={'M': 4096, 'dtype': torch.float32},
    ))
def benchmark_bwd(M, N, dtype, provider):
    x = torch.randn((M, N), device=DEVICE, dtype=dtype, requires_grad=True)
    gy = torch.randn((M, N), device=DEVICE, dtype=dtype)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    # Build the autograd graph once outside the timed loop, then only time the
    # backward pass itself via torch.autograd.grad with retain_graph=True.
    if provider == 'torch':
        y = torch.softmax(x, dim=1)
    else:
        y = OnlineSoftmaxFunc.apply(x)

    ms = triton.testing.do_bench(
        lambda: torch.autograd.grad(y, x, gy, retain_graph=True)
    )
    # Bytes moved: read gy, read y, write dx -> 3 * M * N * sizeof(dtype).
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark_fwd.run(show_plots=True, print_data=True)
benchmark_bwd.run(show_plots=True, print_data=True)
# %%
