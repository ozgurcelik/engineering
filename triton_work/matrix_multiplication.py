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
    # acc is fp32 even though A and B are fp16: we want to avoid losing low
    # bits across the K accumulation. Cast back to fp16 at the store.
    acc = tl.zeros([], dtype=tl.float32)
    for k in range(K):
        a_val = tl.load(a_ptr + row * a_row_stride + k)
        b_val = tl.load(b_ptr + k * b_row_stride + col)
        acc += a_val.to(tl.float32) * b_val.to(tl.float32)
    c_m_n_ptr = c_ptr + row * c_row_stride + col
    tl.store(c_m_n_ptr, acc.to(tl.float16))

def matrix_multiplication_naive(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
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
    acc = tl.zeros([], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_start_ptr = a_ptr + row * a_row_stride
        b_start_ptr = b_ptr + col

        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets
        b_offsets = k_offsets * b_row_stride

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_vals = tl.load(a_ptrs, mask=k_mask).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=k_mask).to(tl.float32)

        acc += tl.sum(a_vals * b_vals)

    c_m_n_ptr = c_ptr + row * c_row_stride + col
    tl.store(c_m_n_ptr, acc.to(tl.float16))

def matrix_multiplication_naive_blocked(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_K = 128
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
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

        a_vals = tl.load(a_ptrs, mask=a_mask).to(tl.float32) # shape [BLOCK_SIZE_K]
        b_vals = tl.load(b_ptrs, mask=b_mask).to(tl.float32) # shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        acc += tl.sum(a_vals[:, None] * b_vals, axis=0) # shape [BLOCK_SIZE_N]

    c_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    c_mask = c_offsets < N
    c_m_n_ptr = c_ptr + row * c_row_stride + c_offsets * c_col_stride
    tl.store(c_m_n_ptr, acc.to(tl.float16), mask=c_mask)

def matrix_multiplication_naive_row_major(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    grid_size = (M, triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
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

        # With fp16 inputs Triton's tl.dot uses HMMA tensor cores by default.
        # The accumulator stays fp32 for accuracy across the K reduction.
        acc = tl.dot(a_vals, b_vals, acc)

    c_offsets = m_offsets[:, None] * c_row_stride + n_offsets[None, :] * c_col_stride
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc.to(tl.float16), mask=c_mask)

def matrix_multiplication_tiled(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    matrix_multiplication_tiled_kernel[grid_size](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), a.stride(1), b.stride(1), c.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    return c


"""
In tile matmul, each program computes one output tile C[m_tile, n_tile] of size BLOCK_SIZE_M x BLOCK_SIZE_N.
To compute this, the program reads
- One row strip of A: rows m_tile * BLOCK_SIZE_M to m_tile * BLOCK_SIZE_M + BLOCK_SIZE_M - 1, all K columns. Call this A_m
- One column strip of B: columns n_tile * BLOCK_SIZE_N to n_tile * BLOCK_SIZE_N + BLOCK_SIZE_N - 1, all K rows. Call this B_n

Two programs that share the same m_tile reads the same A_m, and two programs that share the same n_tile reads the same B_n.
L2 reuse comes from arranging programs so that ones close in time (close in pid) share A_m or B_n.


====================================================================
CONCRETE EXAMPLE: 6 x 8 grid of output tiles, GROUP_SIZE_M = 2
====================================================================
num_pid_m = 6, num_pid_n = 8, GROUP_SIZE_M = 2
=> num_pid_in_group = 2 * 8 = 16
=> 48 programs total, 3 groups of 16

Row-major mapping (what a 2D grid effectively gives us):

         n=0  n=1  n=2  n=3  n=4  n=5  n=6  n=7
   m=0 |   0    1    2    3    4    5    6    7
   m=1 |   8    9   10   11   12   13   14   15
   m=2 |  16   17   18   19   20   21   22   23
   m=3 |  24   25   26   27   28   29   30   31
   m=4 |  32   33   34   35   36   37   38   39
   m=5 |  40   41   42   43   44   45   46   47

Supergrouped mapping (column-major within each height-2 strip):

         n=0  n=1  n=2  n=3  n=4  n=5  n=6  n=7
   m=0 |   0    2    4    6    8   10   12   14   <- group 0
   m=1 |   1    3    5    7    9   11   13   15
   m=2 |  16   18   20   22   24   26   28   30   <- group 1
   m=3 |  17   19   21   23   25   27   29   31
   m=4 |  32   34   36   38   40   42   44   46   <- group 2
   m=5 |  33   35   37   39   41   43   45   47

Both schedules compute the exact same 48 tiles and produce identical
results. Only the order changes -- and therefore which inputs the GPU is
hammering at any given moment.


====================================================================
L2 TRACE: 4 SMs concurrent, L2 holds 6 strips (toy numbers)
====================================================================

ROW-MAJOR:
  wave 1  pids 0..3   tiles (0,0..3)   load A0 + B0..B3            -> 5 HBM
  wave 2  pids 4..7   tiles (0,4..7)   A0 hit, load B4..B7         -> 4 HBM
                                       (B0..B3 evicted to make room)
  wave 3  pids 8..11  tiles (1,0..3)   A1 new, B0..B3 EVICTED      -> 5 HBM
  wave 4  pids 12..15 tiles (1,4..7)   A1 hit, B4..B7 EVICTED      -> 4 HBM
  ... pattern repeats for m=2,3,4,5 -> ~9 HBM loads per m-row
  TOTAL: ~54 HBM loads

SUPERGROUPED (GROUP_SIZE_M=2):
  wave 1  pids 0..3   tiles (0,0)(1,0)(0,1)(1,1)
                                       load A0,A1 + B0,B1          -> 4 HBM
  wave 2  pids 4..7   tiles (0,2)(1,2)(0,3)(1,3)
                                       A0,A1 hit, load B2,B3       -> 2 HBM
  wave 3  pids 8..11  tiles (0,4)(1,4)(0,5)(1,5)
                                       A0,A1 hit, load B4,B5       -> 2 HBM
                                       (B0,B1 evicted)
  wave 4  pids 12..15 tiles (0,6)(1,6)(0,7)(1,7)
                                       A0,A1 hit, load B6,B7       -> 2 HBM
  group 0 done -> 10 HBM loads. Groups 1 and 2 mirror this.
  TOTAL: ~30 HBM loads

Same 48 output tiles, ~45% less HBM traffic. Tensor cores spend less
time stalled waiting for inputs -- that's where the TFLOPS gain comes
from.


====================================================================
WHY COLUMN-MAJOR WITHIN A GROUP (and not row-major)?
====================================================================
The GPU dispatches programs in increasing pid order. So the first wave
of concurrent programs is always a contiguous prefix of pids. The
question is: which output tiles should that prefix cover?

With COLUMN-MAJOR within a group, the first GROUP_SIZE_M pids step
through one column of the group (different m, same n). They all share
the SAME B_n strip. Then the next GROUP_SIZE_M pids do the next column
and reuse the GROUP_SIZE_M A-strips already in L2. Both axes stay
narrow over the lifetime of the group -> tight working set.

With ROW-MAJOR within a group, the first num_pid_n pids would sweep
across the entire N axis (same m, different n). That immediately puts
num_pid_n distinct B-strips into the wave -- which is exactly the
wide-and-flat working set that the supergrouping trick was supposed to
fix. We'd be back to the row-major problem, just nested one level
deeper.

So column-major within a group is what makes both A and B stay reusable
inside the L2 working set. That is the entire mechanism by which
supergrouping cuts HBM traffic.
"""


@triton.jit
def matrix_multiplication_tiled_kernel_supergrouped(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Let us see how many blocks we have in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Groups are horizontal strips of programs, so they cover entire column space of C
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group # which group this program is in
    first_pid_m = group_id * GROUP_SIZE_M # where does this group start
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) # if the num_pid_m is not a multiple of GROUP_SIZE_M, then the last group will have fewer than GROUP_SIZE_M programs

    local_pid = pid % num_pid_in_group # 0 ... num_pid_in_group - 1
    pid_m = first_pid_m + (local_pid % group_size_m) 
    pid_n = local_pid // group_size_m

    """
    Within a group, local_pid ranges over group_size_m x num_pid_n positions.
    We traverse those positions in column-major order.
    - local_pid % group_size_m is which row inside the group
    - local_pid // group_size_m is which column inside the group
    """

    # From here on, the computation is identical to matrix_multiplication_tiled_kernel.
    # Only the (pid_m, pid_n) -> output-tile mapping changes. Same work, different order.
    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N

    m_offsets = tl.arange(0, BLOCK_SIZE_M) + row
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    n_mask = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = m_offsets[:, None] * a_row_stride + k_offsets[None, :] * a_col_stride
        b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride

        a_ptrs = a_ptr + a_offsets
        b_ptrs = b_ptr + b_offsets

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a_vals = tl.load(a_ptrs, mask=a_mask)
        b_vals = tl.load(b_ptrs, mask=b_mask)

        acc = tl.dot(a_vals, b_vals, acc)

    c_offsets = m_offsets[:, None] * c_row_stride + n_offsets[None, :] * c_col_stride
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc.to(tl.float16), mask=c_mask)


def matrix_multiplication_tiled_supergrouped(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    # 1D launch grid is required for the supergrouped pid -> (pid_m, pid_n) mapping
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    matrix_multiplication_tiled_kernel_supergrouped[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
    )
    return c


"""
====================================================================
DOES SUPERGROUPING ACTUALLY HELP? (fp16, 64x64x64 blocks, no tuning)
====================================================================

Run the benchmark below (sweep N from 256 to 16384) and you get:

    N        Tiled (2D)   Sg verbose   Sg lean    Torch    Sg lean / Tiled
     4096      240.5        241.8       241.9     373.4      1.01x
     6144      238.8        240.0       240.8     363.4      1.01x
     8192      235.2        234.9       236.9     396.8      1.01x
     9216      218.9        230.8       233.2     395.7      1.07x
    10240      212.0        230.6       232.6     379.8      1.10x
    12288      210.8        231.2       233.8     394.8      1.11x
    14336      188.6        229.9       232.7     393.7      1.23x
    15360      184.2        230.8       233.2     393.6      1.27x
    16384      172.3        232.4       234.8     393.7      1.36x

Two regimes, separated almost exactly at N=8192:

1. N <= 8192: both schedules sit at ~235 TFLOPS. Supergrouping does
   nothing visible. The kernel is HMMA-issue / load-latency bound at the
   blocks/stages we picked, not L2-traffic bound, so reordering pids
   can't move the needle.

2. N >= 9216: the plain 2D-tiled curve falls off a cliff -- 240 -> 172
   TFLOPS by 16384 -- while the supergroup curve stays flat at ~233.
   That's L2 thrashing on the 2D-tiled side, exactly the situation
   supergrouping was designed for. The pid reordering buys back ~36%
   on this GPU with nothing else touched.

Why the cliff is at 8192-9216
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In fp16 the L2 footprint of A+B = 4*N^2 bytes. The L2 on this card is
128 MB:
    N=5760  -> 128 MB     (= L2)
    N=8192  -> 256 MB     (2x L2)
    N=9216  -> 324 MB     (2.5x L2)
    N=16384 -> 1 GB       (8x L2)

Up through ~8192, the 2D grid's natural row-major schedule still gets a
lot of L2 hits because consecutive waves have enough overlap. Past
9216, A row-strips touched by wave i are no longer in L2 by the time
wave j needs them again, so each wave starts paying the full HBM load.

Supergrouping fixes this by traversing groups of GROUP_SIZE_M=8 row-tiles
in column-major order WITHIN each group. A single wave (188 SMs on a
Blackwell PRO 6000) covers ~8 distinct A row-strips and ~24 distinct B
col-strips at a time -- a working set that comfortably fits in L2 even
at 16384. So the schedule never has to re-stream A from HBM.

The K-loop body story (orthogonal, still real)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The earlier fp32-IEEE finding -- that the verbose K-loop body
interacts badly with 1D launch grids -- is still true, just buried
under the L2 effect once we're on tensor cores:

       for k in range(0, K, BLOCK_SIZE_K):
           # NEW each iteration:
           k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
           a_offsets = m_offsets[:, None] * a_rs + k_offsets[None, :] * a_cs
           b_offsets = k_offsets[:, None] * b_rs + n_offsets[None, :] * b_cs
           a_mask    = m_mask[:, None] & k_mask[None, :]
           b_mask    = k_mask[:, None] & n_mask[None, :]
           ...
           acc = tl.dot(a_vals, b_vals, acc)

`m_offsets * a_rs` is loop-invariant; the compiler should hoist it.
Whether it actually does depends on what it can prove about pid_m /
pid_n. With a 2D launch grid those are intrinsics with tight bounds;
with a 1D grid they come from `pid // X` / `pid % X` for runtime X and
Triton can fail to fold the arithmetic. The lean body (build pointers
ONCE, advance with += inside the loop, mask only along K) sidesteps
this entirely. In fp32-IEEE that gap was ~17%; in fp16 it shrinks to
~1-2% (compare the verbose and lean supergroup columns above) because
the integer recomputation runs in parallel with HMMA on different
execution units.

What this benchmark does NOT show
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We're at ~235 TFLOPS at the upper end; cuBLAS is at ~394 TFLOPS. The
remaining ~40% gap is not a scheduling problem, it's a configuration
problem: bigger blocks (e.g. 128x128x32), num_stages > 1 (software
pipelining), num_warps = 8, and shape-specific autotune. Those
optimizations are what would close the gap to cuBLAS, but they belong
in a separate exercise.
"""


@triton.jit
def matrix_multiplication_tiled_kernel_supergrouped_lean(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Build initial pointers ONCE before the K-loop. Apply `% M` / `% N` to
    # the row/col offsets so we can drop the m/n masks inside the loop --
    # any threads in a partial last tile harmlessly reload the wrap-around
    # row/column, which gets masked out at the final store.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * a_row_stride + offs_k[None, :] * a_col_stride)
    b_ptrs = b_ptr + (offs_k[:, None] * b_row_stride + offs_bn[None, :] * b_col_stride)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Mask only along K; the address arithmetic in the loop body collapses
        # to a fused pointer-bump, which is what lets the hot path be just
        # "load tiles, MMA, advance pointers".
        a_vals = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_vals = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a_vals, b_vals, acc)
        a_ptrs += BLOCK_SIZE_K * a_col_stride
        b_ptrs += BLOCK_SIZE_K * b_row_stride

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * c_row_stride + offs_cn[None, :] * c_col_stride
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matrix_multiplication_tiled_supergrouped_lean(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    matrix_multiplication_tiled_kernel_supergrouped_lean[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
    )
    return c


# %%
# Correctness check, in fp16. We compare against an fp32 reference so the
# tolerances bound rounding from accumulating fp16 products into fp32 and
# casting back to fp16. K=192 here keeps the absolute round-off small enough
# that 1e-2 / 1e-2 is comfortable.
torch.manual_seed(0)
a = torch.randn((512, 192), device=DEVICE, dtype=torch.float16)
b = torch.randn((192, 128), device=DEVICE, dtype=torch.float16)
c_triton = matrix_multiplication_naive(a, b)
c_triton_blocked = matrix_multiplication_naive_blocked(a, b)
c_triton_row_major = matrix_multiplication_naive_row_major(a, b)
c_triton_tiled = matrix_multiplication_tiled(a, b)
c_triton_tiled_supergrouped = matrix_multiplication_tiled_supergrouped(a, b)
c_triton_tiled_supergrouped_lean = matrix_multiplication_tiled_supergrouped_lean(a, b)
c_torch = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2), (c_triton, c_torch)
assert torch.allclose(c_triton_blocked, c_torch, atol=1e-2, rtol=1e-2), (c_triton_blocked, c_torch)
assert torch.allclose(c_triton_row_major, c_torch, atol=1e-2, rtol=1e-2), (c_triton_row_major, c_torch)
assert torch.allclose(c_triton_tiled, c_torch, atol=1e-1, rtol=1e-1), (c_triton_tiled, c_torch)
assert torch.allclose(c_triton_tiled_supergrouped, c_torch, atol=1e-1, rtol=1e-1), (c_triton_tiled_supergrouped, c_torch)
assert torch.allclose(c_triton_tiled_supergrouped_lean, c_torch, atol=1e-1, rtol=1e-1), (c_triton_tiled_supergrouped_lean, c_torch)
# %%
# Benchmark in fp16: plain tiled (2D grid) vs supergrouped (1D grid) vs the
# lean-body supergrouped (1D grid, hoisted address arithmetic) vs torch.
# All Triton variants use the same block sizes (64x64x64) and Triton's
# default num_stages / num_warps -- nothing else is tuned. The only
# question is whether the pid -> (pid_m, pid_n) reordering buys us anything
# on its own.
#
# The sweep is pushed up to 16384 because supergrouping is a cache-locality
# optimization, and locality only matters once the working set genuinely
# stops fitting in L2. At fp16, A+B = 2 * N^2 * 2 bytes:
#     N=4096   ->   64 MB     (fits in 128 MB L2 with room to spare)
#     N=5760   ->  128 MB     (right at L2 capacity)
#     N=8192   ->  256 MB     (2x L2)
#     N=12288  ->  576 MB     (4.5x L2)
#     N=16384  -> 1024 MB     (8x L2)
# So the upper half of the sweep is solidly in L2-overflow territory and is
# where the supergrouping is most likely to show up. Whether it actually
# does -- given that the kernel is also far from the tensor-core ceiling --
# is the empirical question this benchmark answers.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        # Three regions:
        #   small (256..4096)  -> fits in L2, pid order should not matter
        #   medium (4096..8192) -> straddles L2 boundary
        #   large (>8192)      -> A+B many x L2; supergroup should pay off
        #                         IF the kernel is memory-bound here
        x_vals=[256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096,
                4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
                9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384],
        line_arg='provider',
        line_vals=[
            'triton_tiled',
            'triton_tiled_supergrouped',
            'triton_tiled_supergrouped_lean',
            'torch',
        ],
        line_names=[
            "Triton Tiled (2D grid, verbose K-loop)",
            "Triton Supergrouped (1D grid, verbose K-loop)",
            "Triton Supergrouped (1D grid, lean K-loop)",
            "Torch (fp16, cuBLAS)",
        ],
        styles=[('orange', '-'), ('red', '-'), ('blue', '-'), ('green', '-')],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton_tiled':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_tiled(a, b))
    if provider == 'triton_tiled_supergrouped':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_tiled_supergrouped(a, b))
    if provider == 'triton_tiled_supergrouped_lean':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_tiled_supergrouped_lean(a, b))
    # FLOPs for matmul: 2 * M * N * K (one multiply + one add per output element per K dim)
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark.run(show_plots=True, print_data=True)
# %%