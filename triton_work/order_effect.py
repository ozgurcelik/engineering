# %%
"""
====================================================================
EXPERIMENT: does the `order` arg to tl.make_block_ptr actually matter?
====================================================================

Setup. We want to compute one of the two matmul backward operations:

    C = G @ B^T               # shape: (M, R) @ (R, K) -> (M, K)

where:
  - G is provided as a row-major (M, R) tensor with strides (R, 1).
  - B is provided as a row-major (K, R) tensor with strides (R, 1).
    Note: B has its REDUCTION axis (R) as its inner (contiguous) axis.
    This is the layout that comes out of a typical forward matmul's
    right-hand input -- e.g. in C = A @ B with B shape (K, N), B's
    contiguous axis is N. Here we're reusing B from such a forward
    pass and want B^T, which means viewing B as (N, K) with stride
    (1, N). In this file we call that reduction axis R.

The "stride-swap" trick says: instead of materialising B^T with
b.t().contiguous(), pass B with swapped strides into a matmul kernel
and let the block pointer treat it as transposed at zero memory cost.

That leaves one question for the kernel author: what `order=` do we
pass to tl.make_block_ptr for the transposed view?

  shape   = (R, K)
  strides = (b.stride(1), b.stride(0)) = (1, R)

  -> dim 0 has stride 1   (contiguous in memory)
  -> dim 1 has stride R

The `order` tuple in make_block_ptr lists dims from MOST contiguous
to LEAST. So:

  HONEST  order = (0, 1)        # tells Triton: dim 0 is the fast axis
  LYING   order = (1, 0)        # claims dim 1 is fast, but it isn't

Both produce CORRECT results -- strides are authoritative for
address arithmetic, the `order` tuple is only a hint to the compiler
about which axis to vectorize / which MMA layout to pick. The question
is purely performance: how much does the wrong hint cost?

We benchmark three providers across square sizes (M = R = K = N):
  - honest:     stride-swap + order=(0, 1)
  - lying:      stride-swap + order=(1, 0)
  - contiguous: pre-materialise B.t().contiguous(), then use natural
                NN order=(1, 0). The copy cost is INCLUDED in the
                timing because that's the real-world alternative.

All three kernels are otherwise identical -- same autotune config
list, same supergrouped persistent layout, same block pointers --
so any difference between honest and lying is purely the `order`
hint, and any difference vs contiguous is the cost of the memory copy.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count


# A focused config list. We want first-call autotune latency to stay
# reasonable across three kernels, so we pick a small set of plausible
# configs spanning compute-bound and memory-bound regimes. These mirror
# the "winners" from matrix_multiplication.py's broader sweep.
def get_autotune_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_R': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_R': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_R': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_R': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_R': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
    ]


# Kernel template. Computes C = G @ B^T using block pointers and the
# supergrouped + persistent launch pattern from matrix_multiplication.py.
# The ONLY difference between the two stride-swap variants is the `order`
# argument on the B^T block pointer; everything else is held constant.
#
# Convention used inside this kernel:
#   M  = rows of output C (and rows of G)
#   K  = cols of output C (and "rows" of B^T view, = cols of underlying B)
#   R  = reduction axis    (cols of G, = "cols" of B^T view, = rows of underlying B)
#
# Inputs to the kernel are passed in B^T VIEW coordinates. The launcher
# is responsible for the stride swap on B before calling.
@triton.autotune(configs=get_autotune_configs(), key=['M', 'K', 'R'])
@triton.jit
def matmul_gbt_honest_kernel(
    g_ptr, bt_ptr, c_ptr,
    M, K, R,
    g_stride_m, g_stride_r,
    bt_stride_r, bt_stride_k,   # B viewed transposed: (R, K) with strides (1, R)
    c_stride_m, c_stride_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_start = tl.program_id(0)
    p_count = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_total = num_pid_m * num_pid_k
    num_pid_in_group = GROUP_SIZE_M * num_pid_k

    tl.assume(g_stride_m > 0)
    tl.assume(g_stride_r > 0)
    tl.assume(bt_stride_r > 0)
    tl.assume(bt_stride_k > 0)
    tl.assume(c_stride_m > 0)
    tl.assume(c_stride_k > 0)

    for pid in range(pid_start, num_pid_total, p_count):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        local_pid = pid % num_pid_in_group
        pid_m = first_pid_m + (local_pid % group_size_m)
        pid_k = local_pid // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_k >= 0)

        # G: row-major (M, R) -> dim 1 (R) is contiguous -> order=(1, 0) is truthful.
        g_block_ptr = tl.make_block_ptr(
            g_ptr,
            shape=(M, R),
            strides=(g_stride_m, g_stride_r),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_R),
            order=(1, 0),
        )

        # B^T view: shape (R, K), strides (1, R) -> dim 0 is contiguous.
        # HONEST: order=(0, 1) tells Triton the truth about which dim is fast.
        bt_block_ptr = tl.make_block_ptr(
            bt_ptr,
            shape=(R, K),
            strides=(bt_stride_r, bt_stride_k),
            offsets=(0, pid_k * BLOCK_SIZE_K),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_K),
            order=(0, 1),
        )

        # C: row-major (M, K) -> dim 1 (K) is contiguous.
        c_block_ptr = tl.make_block_ptr(
            c_ptr,
            shape=(M, K),
            strides=(c_stride_m, c_stride_k),
            offsets=(pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
        for _ in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
            g_vals = tl.load(g_block_ptr, boundary_check=(0, 1), padding_option="zero")
            bt_vals = tl.load(bt_block_ptr, boundary_check=(0, 1), padding_option="zero")

            acc = tl.dot(g_vals, bt_vals, acc)

            g_block_ptr = tl.advance(g_block_ptr, (0, BLOCK_SIZE_R))
            bt_block_ptr = tl.advance(bt_block_ptr, (BLOCK_SIZE_R, 0))

        tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Identical to the honest kernel above EXCEPT for order=(1, 0) on the B^T
# block pointer. This is what you get if you reuse your forward matmul
# kernel as-is (with its hardcoded order=(1, 0)) and just pass B with
# swapped strides -- the strides drive correctness, but the `order` hint
# now misdescribes the actual memory layout.
@triton.autotune(configs=get_autotune_configs(), key=['M', 'K', 'R'])
@triton.jit
def matmul_gbt_lying_kernel(
    g_ptr, bt_ptr, c_ptr,
    M, K, R,
    g_stride_m, g_stride_r,
    bt_stride_r, bt_stride_k,
    c_stride_m, c_stride_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_start = tl.program_id(0)
    p_count = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_total = num_pid_m * num_pid_k
    num_pid_in_group = GROUP_SIZE_M * num_pid_k

    tl.assume(g_stride_m > 0)
    tl.assume(g_stride_r > 0)
    tl.assume(bt_stride_r > 0)
    tl.assume(bt_stride_k > 0)
    tl.assume(c_stride_m > 0)
    tl.assume(c_stride_k > 0)

    for pid in range(pid_start, num_pid_total, p_count):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        local_pid = pid % num_pid_in_group
        pid_m = first_pid_m + (local_pid % group_size_m)
        pid_k = local_pid // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_k >= 0)

        g_block_ptr = tl.make_block_ptr(
            g_ptr,
            shape=(M, R),
            strides=(g_stride_m, g_stride_r),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_R),
            order=(1, 0),
        )

        # LYING: order=(1, 0) claims dim 1 is contiguous, but for the
        # stride-swapped B^T view it's actually dim 0. Strides are still
        # authoritative for correctness; the lie only affects the compiler's
        # layout choices.
        bt_block_ptr = tl.make_block_ptr(
            bt_ptr,
            shape=(R, K),
            strides=(bt_stride_r, bt_stride_k),
            offsets=(0, pid_k * BLOCK_SIZE_K),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_K),
            order=(1, 0),
        )

        c_block_ptr = tl.make_block_ptr(
            c_ptr,
            shape=(M, K),
            strides=(c_stride_m, c_stride_k),
            offsets=(pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
        for _ in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
            g_vals = tl.load(g_block_ptr, boundary_check=(0, 1), padding_option="zero")
            bt_vals = tl.load(bt_block_ptr, boundary_check=(0, 1), padding_option="zero")

            acc = tl.dot(g_vals, bt_vals, acc)

            g_block_ptr = tl.advance(g_block_ptr, (0, BLOCK_SIZE_R))
            bt_block_ptr = tl.advance(bt_block_ptr, (BLOCK_SIZE_R, 0))

        tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Baseline: pre-materialise B^T with .contiguous() and then call a plain
# NN matmul. This is the "Option 3 strawman" -- it pays a full memory
# copy of B before each call. The kernel here is exactly the standard
# NN matmul from matrix_multiplication.py, included verbatim so the only
# difference vs the two above is the Python-side contiguous() copy.
@triton.autotune(configs=get_autotune_configs(), key=['M', 'K', 'R'])
@triton.jit
def matmul_nn_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, R,
    a_stride_m, a_stride_r,
    b_stride_r, b_stride_k,
    c_stride_m, c_stride_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_start = tl.program_id(0)
    p_count = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_total = num_pid_m * num_pid_k
    num_pid_in_group = GROUP_SIZE_M * num_pid_k

    tl.assume(a_stride_m > 0)
    tl.assume(a_stride_r > 0)
    tl.assume(b_stride_r > 0)
    tl.assume(b_stride_k > 0)
    tl.assume(c_stride_m > 0)
    tl.assume(c_stride_k > 0)

    for pid in range(pid_start, num_pid_total, p_count):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        local_pid = pid % num_pid_in_group
        pid_m = first_pid_m + (local_pid % group_size_m)
        pid_k = local_pid // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_k >= 0)

        a_block_ptr = tl.make_block_ptr(
            a_ptr,
            shape=(M, R),
            strides=(a_stride_m, a_stride_r),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_R),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            b_ptr,
            shape=(R, K),
            strides=(b_stride_r, b_stride_k),
            offsets=(0, pid_k * BLOCK_SIZE_K),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_K),
            order=(1, 0),
        )
        c_block_ptr = tl.make_block_ptr(
            c_ptr,
            shape=(M, K),
            strides=(c_stride_m, c_stride_k),
            offsets=(pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
        for _ in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
            a_vals = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
            b_vals = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")

            acc = tl.dot(a_vals, b_vals, acc)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_R))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_R, 0))

        tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


def _persistent_grid(M, K):
    return lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),
        ),
    )


def matmul_gbt_honest(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = G @ B^T via stride-swap with HONEST order=(0,1) on the B^T view."""
    M, R = g.shape
    K, R2 = b.shape
    assert R == R2, f"reduction dims must match: g has R={R}, b has R={R2}"
    c = torch.empty((M, K), device=DEVICE, dtype=g.dtype)
    matmul_gbt_honest_kernel[_persistent_grid(M, K)](
        g, b, c,
        M, K, R,
        g.stride(0), g.stride(1),
        b.stride(1), b.stride(0),   # <-- THE STRIDE SWAP. b passed as B^T.
        c.stride(0), c.stride(1),
    )
    return c


def matmul_gbt_lying(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = G @ B^T via stride-swap with LYING order=(1,0) on the B^T view."""
    M, R = g.shape
    K, R2 = b.shape
    assert R == R2
    c = torch.empty((M, K), device=DEVICE, dtype=g.dtype)
    matmul_gbt_lying_kernel[_persistent_grid(M, K)](
        g, b, c,
        M, K, R,
        g.stride(0), g.stride(1),
        b.stride(1), b.stride(0),
        c.stride(0), c.stride(1),
    )
    return c


def matmul_gbt_contiguous(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = G @ B^T via b.t().contiguous() + natural NN matmul.

    The contiguous() copy is INSIDE the timed region. That's the real-world
    cost of avoiding the stride-swap trick.
    """
    M, R = g.shape
    K, R2 = b.shape
    assert R == R2
    bt = b.t().contiguous()              # <-- full memory copy: R * K * dtype bytes
    c = torch.empty((M, K), device=DEVICE, dtype=g.dtype)
    matmul_nn_kernel[_persistent_grid(M, K)](
        g, bt, c,
        M, K, R,
        g.stride(0), g.stride(1),
        bt.stride(0), bt.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# %%
# Correctness check. All three Triton variants should agree with torch's
# g @ b.t() reference to within fp16 accumulation tolerance.
torch.manual_seed(0)
M, R, K = 512, 192, 384
g = torch.randn((M, R), device=DEVICE, dtype=torch.float16)
b = torch.randn((K, R), device=DEVICE, dtype=torch.float16)

c_ref = (g.to(torch.float32) @ b.t().to(torch.float32)).to(torch.float16)
c_honest = matmul_gbt_honest(g, b)
c_lying = matmul_gbt_lying(g, b)
c_contig = matmul_gbt_contiguous(g, b)

assert torch.allclose(c_honest, c_ref, atol=1e-1, rtol=1e-1), (c_honest, c_ref)
assert torch.allclose(c_lying, c_ref, atol=1e-1, rtol=1e-1), (c_lying, c_ref)
assert torch.allclose(c_contig, c_ref, atol=1e-1, rtol=1e-1), (c_contig, c_ref)
print("OK: all three variants agree with torch reference.")


# %%
# Benchmark. Square shapes M = R = K = N so the FLOP count is 2*N^3 and
# we can directly compare TFLOPS across sizes. Same sweep as
# matrix_multiplication.py's benchmark so the absolute TFLOPS numbers
# are roughly comparable to the readings in that file's comment block.
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 1024, 2048, 3072, 4096,
                5120, 6144, 7168, 8192,
                9216, 10240, 12288, 14336, 16384],
        line_arg='provider',
        line_vals=['honest', 'lying', 'contiguous', 'torch'],
        line_names=[
            "stride-swap, order=(0,1) [HONEST]",
            "stride-swap, order=(1,0) [LYING]",
            "b.t().contiguous() + NN matmul",
            "Torch (cuBLAS, fp16)",
        ],
        styles=[('blue', '-'), ('orange', '-'), ('purple', '--'), ('green', '-')],
        ylabel="TFLOPS",
        plot_name="order-effect-on-stride-swapped-matmul",
        args={},
    ))
def benchmark(N, provider):
    g = torch.randn((N, N), device=DEVICE, dtype=torch.float16)
    b = torch.randn((N, N), device=DEVICE, dtype=torch.float16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'honest':
        ms = triton.testing.do_bench(lambda: matmul_gbt_honest(g, b))
    elif provider == 'lying':
        ms = triton.testing.do_bench(lambda: matmul_gbt_lying(g, b))
    elif provider == 'contiguous':
        ms = triton.testing.do_bench(lambda: matmul_gbt_contiguous(g, b))
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: g @ b.t())
    # 2 * M * K * R FLOPs; here M = K = R = N, so 2 * N^3.
    tflops = lambda ms: 2 * N * N * N * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark.run(show_plots=True, print_data=True)


"""
====================================================================
RESULTS (NVIDIA L40S, Ada Lovelace, 142 SMs, fp16)
Triton 3.6.0, torch 2.11.0+cu130
====================================================================

      N      honest    lying    contig    torch     lying/honest
    -----   -------   ------   -------   -------   -------------
      512     24.03    25.00    14.53     28.06       1.04x
     1024     96.10    93.41    45.69    118.12       0.97x
     2048    175.51   171.23   111.09    164.23       0.98x
     3072    164.36   164.36   124.33    166.29       1.00x
     4096    208.24   206.94   164.34    213.63       0.99x
     5120    214.43   215.25   178.16    205.24       1.00x
     6144    195.64   188.04   172.06    188.68       0.96x
     7168    189.25   190.31   172.43    187.58       1.01x
     8192    189.74   192.77   173.44    183.31       1.02x
     9216    183.24   181.82   168.55    175.82       0.99x
    10240    180.73   183.87   169.75    179.14       1.02x
    12288    172.98   185.47   174.59    178.93       1.07x
    14336    179.61   177.90   165.15    177.45       0.99x
    16384    177.54   180.15   173.34    173.59       1.01x

(All numbers in TFLOPS. `lying/honest` is the ratio for the
stride-swap pair only.)

Three things jump out:

1. HONEST vs LYING is, in practice, INDISTINGUISHABLE on this GPU.
   The ratio stays in [0.96, 1.07] across the whole sweep, which is
   inside run-to-run noise of triton.testing.do_bench. There is no
   consistent winner: lying edges ahead at N=8192 and 12288, honest
   edges ahead at N=6144, the rest are flat.

   The earlier prediction was "5-15% penalty for lying on modern
   GPUs". On Ada Lovelace with Triton 3.6 the actual penalty is
   essentially zero. The compiler reads the strides, sees that dim 0
   is the contiguous axis regardless of what the `order` hint says,
   and emits the same MMA layout either way. Strides are
   authoritative for correctness AND, on this stack, sufficient for
   the compiler to pick the right vectorization.

   This is not a license to ignore `order` -- on older GPUs (V100/T4),
   and on workloads with more exotic strides, the hint can still
   matter. But the takeaway for "should I refactor my forward
   kernel to thread `order` through as a constexpr?" is: not for
   perf reasons alone. Refactor only if it makes the code clearer.

2. b.t().contiguous() IS A REAL COST, ESPECIALLY AT SMALL N.
   The contiguous-copy path is 30-50% slower at N <= 1024 and still
   5-15% slower at large N. At N=512 it's nearly HALF the throughput
   of stride-swap. The copy itself is K * R * 2 bytes (fp16) of HBM
   read + HBM write -- bandwidth-bound and not free.

   So if a forward kernel currently hardcodes order=(1,0) and the
   choice is between "leave it and lie" or "leave it and .contiguous()
   the operand first", lying is clearly the right call.

3. STRIDE-SWAPPED MATMULS ARE COMPETITIVE WITH cuBLAS HERE.
   At N=4096-8192 the Triton kernels (honest or lying) match or
   exceed torch's cuBLAS. That's a sanity check that this experiment
   is in a meaningful operating regime, not a kernel so far from the
   tensor-core ceiling that all variants look the same.

====================================================================
PRACTICAL RECOMMENDATION
====================================================================

For the matmul-backward use case discussed in block_pointers.py:

  dA = grad_output @ B^T     # stride-swap B
  dB = A^T @ grad_output     # stride-swap A

Just call the existing forward kernel with stride-swapped operands.
Do NOT bother threading `order` through as a constexpr -- the
measurable speedup on modern GPUs is zero, and the extra parameter
adds complexity to a kernel that's already busy. And definitely do
NOT .contiguous() the transposed operand first; that pays a full
copy for no benefit.

CAVEAT FOR LEARNERS
-------------------
The "zero penalty for lying" result above is specific to:
  - Triton 3.6 on Ada Lovelace
  - Block pointers (which use the `order` hint cooperatively with
    strides; the modern Triton pipeline mostly ignores `order` when
    strides already imply the right layout)
  - Square fp16 GEMMs

If you switch to:
  - manual a_ptrs / b_ptrs index arithmetic instead of make_block_ptr
  - older GPU architectures (V100/T4) without cooperative TMA
  - rectangular shapes where the "wrong" axis is the long one
the gap can reopen. Re-run the experiment on your target hardware
before extrapolating.
"""
# %%
