#%%
import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# Number of SMs on the active GPU. The persistent kernel below caps its
# launch grid at this value: launching more programs than SMs only adds
# scheduling overhead, since the persistent loop already lets each
# program walk through as many output tiles as it needs.
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count
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

Same 48 output tiles, ~45% less HBM (High-Bandwidth Memory, GPUs main DRAM) traffic. 
Tensor cores spend less time stalled waiting for inputs -- that's where the TFLOPS gain comes from.


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

    N         Tiled (2D)    Supergrouped (1D)    Torch     Sg / Tiled
     4096       240.5            241.8           373.4       1.01x
     6144       238.8            240.0           363.4       1.01x
     8192       235.2            234.9           396.8       1.00x
     9216       218.9            230.8           395.7       1.05x
    10240       212.0            230.6           379.8       1.09x
    12288       210.8            231.2           394.8       1.10x
    14336       188.6            229.9           393.7       1.22x
    15360       184.2            230.8           393.6       1.25x
    16384       172.3            232.4           393.7       1.35x

Two regimes, separated almost exactly at N=8192:

1. N <= 8192: both schedules sit at ~235 TFLOPS. Supergrouping does
   nothing visible. The kernel is HMMA-issue / load-latency bound at
   the blocks we picked, not L2-traffic bound, so reordering pids
   can't move the needle.

2. N >= 9216: the plain 2D-tiled curve falls off a cliff -- 240 -> 172
   TFLOPS by 16384 -- while the supergroup curve stays flat at ~232.
   That's L2 thrashing on the 2D-tiled side, exactly the situation
   supergrouping was designed for. The pid reordering buys back ~35%
   on this GPU with nothing else touched.

Aside: tensor-core MMA instruction families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The "MMA" family of SASS instructions all execute one tile of
D = A*B + C on the tensor cores; they differ only in operand dtype:
- HMMA: half-precision (FP16/BF16) matrix multiply-accumulate. This
  is what a Triton tl.dot on FP16 inputs lowers to on Ampere/Hopper.
- IMMA: integer (INT8/INT4) variant.
- DMMA: double-precision (FP64) variant.
- QMMA: FP8 variant on Hopper.

Why the cliff is at 8192-9216
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reuse-critical quantity is B alone. Under the natural row-major
schedule, B's reuse distance is one full sweep across N: tile (r, j)
loads B[:, j*BN:(j+1)*BN], and the same col-strip is needed again at
tile (r+1, j) -- after ~2*N^2 bytes of intervening B traffic (one full
sweep loads every B col-strip exactly once = sizeof(B)). For B to
survive that distance in L2 we need

    sizeof(B) = 2*N^2 bytes  <=  L2

The L2 on this card is 128 MB, so the threshold is N <= 8192:

    N=4096  -> sizeof(B) =  32 MB    (0.25x L2)
    N=8192  -> sizeof(B) = 128 MB    (1.0x L2)
    N=9216  -> sizeof(B) = 162 MB    (1.27x L2)
    N=16384 -> sizeof(B) = 512 MB    (4.0x L2)

A doesn't enter the criterion. Its row-strip is loaded once and reused
by every column tile of one C-row back-to-back, then never touched
again -- short reuse interval, and the strip itself is small
(BLOCK_M*N*2 = 2 MB at N=16384). It fits trivially.

Above N=8192, B can't survive cross-row reuse, so every new C-row
re-streams all of B from HBM. As a side effect, the ~512 MB of B
streaming traffic during one C-row at N=16384 also evicts the in-use
A row-strip mid-row, so A starts re-fetching too -- but the trigger
is purely B no longer fitting.

Supergrouping fixes this by traversing groups of GROUP_SIZE_M=8 row
tiles in column-major order WITHIN each group. A single wave (188 SMs
on a Blackwell PRO 6000) covers ~8 distinct A row-strips and ~24
distinct B col-strips at a time -- a working set that comfortably fits
in L2 even at 16384. So the schedule never has to re-stream A from
HBM.

Where 8 and 24 come from:
- 8 = GROUP_SIZE_M, set literally in the launcher. Inside a group,
  pids are laid out column-major, so 8 consecutive pids form one
  vertical column of C-tiles -> 8 distinct A row-strips, 1 B col-strip.
- 24 ~= 188 SMs / 8-tall-column = 23.5. A wave of 188 concurrent
  programs therefore spans ~24 such columns side by side, sharing the
  same 8 A row-strips across all of them.

Working set at N=16384: each strip is 64 * N * 2 = 2 MB, so
8 A + 24 B = (8+24) * 2 MB = 64 MB -- half of the 128 MB L2. The plain
row-major wave at the same N would touch 1 A row-strip + 188 B
col-strips ~= 378 MB, ~3x oversubscribed. Supergrouping trades a
slightly wider A footprint (1 -> 8) for a much narrower B footprint
(188 -> 24); each B col-strip a program loads is reused 7 more times
by the rest of its column before being allowed to leave L2. Larger N
benefits from larger GROUP_SIZE_M for the same reason -- see autotune
section below.

What this benchmark does NOT show
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We're at ~232 TFLOPS at the upper end; cuBLAS is at ~394 TFLOPS. The
remaining ~40% gap is not a scheduling problem, it's a configuration
problem: bigger blocks (e.g. 128x128x32), num_stages > 1 (software
pipelining), num_warps = 8, and shape-specific autotune. Those
optimizations are what would close the gap to cuBLAS, but they belong
in a separate exercise.
"""


"""
====================================================================
AUTOTUNING
====================================================================

The supergrouped kernel above hard-codes BLOCK_SIZE_M = BLOCK_SIZE_N =
BLOCK_SIZE_K = 64, GROUP_SIZE_M = 8, and inherits Triton's defaults
for num_warps and num_stages (num_warps=4, num_stages=1 here). None of
those choices are optimal across all shapes:

- Small/medium N is compute-bound. The bottleneck is keeping the
  tensor cores fed without bubbles. Bigger output tiles (more work per
  program) and num_stages >= 3 (software-pipelined K-loop -- load
  stage k+1 while computing stage k) help most.
- Large N is L2-bound (the regime the supergrouping fixed). Smaller
  blocks plus a larger GROUP_SIZE_M can shrink the per-wave working
  set even further.

There is no single (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
GROUP_SIZE_M, num_warps, num_stages) that wins both regimes. So we
hand Triton a list of plausible configurations and let it benchmark
each one on the actual shape:

@triton.autotune(configs=[...], key=['M', 'N', 'K'])
@triton.jit
def kernel(...): ...

Mechanics:
- Each triton.Config bundles tl.constexpr values (block sizes,
  GROUP_SIZE_M) with compilation knobs (num_warps, num_stages).
- The `key` argument names runtime arguments whose values, when
  changed, invalidate the cache and trigger a fresh search. (M,N,K)
  is correct here: a new shape gets re-tuned once, then cached.
- The launch grid becomes a callable `lambda META: ...` because the
  caller no longer knows BLOCK_SIZE_*. Triton invokes the lambda with
  the chosen Config's constexpr dict.
- First call with a new (M,N,K) compiles and times every config;
  expect tens of seconds of warm-up. Subsequent calls reuse the cached
  winner and pay no overhead.

Caveats:
- Not every (BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps) is legal for
  tl.dot; illegal combinations are pruned by the autotuner rather
  than crashing.
- A long config list multiplies first-call latency. Keep it focused.
- `tl.assume(...)` is a free perf hint: it tells the integer analysis
  pass that ids/strides are non-negative so address arithmetic can
  drop sign handling.
"""


def get_autotune_configs():
    # The two compilation knobs that aren't constexprs in the kernel signature:
    # - num_warps: how many warps of 32 threads share one program/output tile.
    #   More warps spreads tl.dot across more tensor-core lanes and hides
    #   instruction latency, at the cost of registers per warp.
    # - num_stages: how many K-loop iterations are software-pipelined.
    #   num_stages=3 means while the tensor cores compute on stage k, the loads
    #   for k+1 are issued and k+2's loads are arriving in shared memory. This
    #   is the single biggest reason the default-config kernel above sits at
    #   ~232 TFLOPS vs cuBLAS at ~394.
    return [
        # Compute-bound regime: large output tiles + deep pipelining.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Memory-bound regime (large N): smaller tiles with a wider GROUP_SIZE_M
        # shrink the per-wave L2 working set further than the supergrouped
        # default could.
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
    ]


# Two decorators, in this order: triton.autotune wraps triton.jit. The
# outer decorator is what the call site actually invokes; it picks a
# Config, then forwards into the JIT'd inner kernel with the constexprs
# from that Config injected.
@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_autotuned_kernel(
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

    local_pid = pid % num_pid_in_group
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    # Free perf hints: the integer-analysis pass uses these to drop sign
    # handling on address arithmetic.
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(a_row_stride > 0)
    tl.assume(a_col_stride > 0)
    tl.assume(b_row_stride > 0)
    tl.assume(b_col_stride > 0)
    tl.assume(c_row_stride > 0)
    tl.assume(c_col_stride > 0)

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


def matrix_multiplication_autotuned(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    # The grid depends on BLOCK_SIZE_M and BLOCK_SIZE_N, which the autotuner
    # picks. So the grid is a callable that receives the chosen Config's
    # constexpr dict as `META` and returns the actual launch shape.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    # Note: BLOCK_SIZE_*, GROUP_SIZE_M are NOT passed here -- the autotuner
    # injects them from the winning Config.
    matrix_multiplication_autotuned_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
    )
    return c


"""
====================================================================
POINTER ADVANCEMENT + MODULO-TRICK MASKING
====================================================================

Two cleanups on top of the autotuned kernel, both copied from the
tutorial in 03-matrix-multiplication.py.

(1) Pointer advancement instead of pointer recomputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The autotuned kernel rebuilds a_offsets/b_offsets/a_ptrs/b_ptrs from
scratch on every K iteration -- a 2D outer-product of m_offsets and
k_offsets (and similarly for B). The compiler often CSEs that, but
not always; the safer pattern is to compute the initial pointer
blocks ONCE before the loop, then increment by a 1D offset every
iteration:

    a_ptrs += BLOCK_SIZE_K * a_col_stride   # shifts entire block right
    b_ptrs += BLOCK_SIZE_K * b_row_stride   # shifts entire block down

Each iteration now does one elementwise add of a scalar onto a
[M, K] block of pointers instead of regenerating the whole 2D
address grid. Modest cycle savings in the inner loop -- exactly
where they matter.

(2) Modulo-trick for M/N boundary handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The autotuned kernel paid for explicit m_mask, n_mask, k_mask AND
combined them into a_mask = m_mask & k_mask, b_mask = k_mask & n_mask.
That's a per-iteration 2D predicate computation plus a per-element
predicate evaluation inside tl.load.

The tutorial trick: clamp M/N out-of-bounds offsets back into bounds
with `% M` / `% N`. Out-of-bounds rows of A wrap to in-bounds rows
(garbage data from the kernel's perspective), out-of-bounds cols of
B wrap to in-bounds cols (also garbage), and the dot product computes
garbage values for rows/cols past M/N -- but those output positions
are then DISCARDED by the c_mask at the final tl.store. Net result:
correct output, no per-iteration M/N masking in the hot loop.

The K dimension still needs a real mask because contributions from
out-of-bounds K elements get summed into VALID output rows, and that
WOULD corrupt the result. The mask `offs_k < K - k * BLOCK_SIZE_K`
zeros out those contributions. Only the last iteration ever has any
masked elements, so the cost is one cheap predicate per iteration
that the tensor cores can usually overlap with the main load.
"""


@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_pointer_kernel(
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

    local_pid = pid % num_pid_in_group
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(a_row_stride > 0)
    tl.assume(a_col_stride > 0)
    tl.assume(b_row_stride > 0)
    tl.assume(b_col_stride > 0)
    tl.assume(c_row_stride > 0)
    tl.assume(c_col_stride > 0)

    # Modulo trick: out-of-bounds row/col indices wrap into bounds. The data
    # loaded for those positions is garbage from the math's POV, but the
    # corresponding output positions are discarded by the c_mask at the
    # final tl.store -- so the computed garbage never escapes the kernel.
    # Maps to old's `m_offsets`/`n_offsets` + `m_mask`/`n_mask`, but the
    # mask is replaced by the wrap and hoisted out of the K loop entirely.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # Note: NO `+ k` here. Old version did `k_offsets = tl.arange(BSK) + k`
    # inside the loop; the `+ k` shift now lives in the pointer advance below.
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initial pointer blocks. Built once; then we only ever ADVANCE them.
    # Old version rebuilt this 2D outer product (m_offsets[:,None]*stride +
    # k_offsets[None,:]*stride) on every K iteration -- we do it just once.
    a_ptrs = a_ptr + (offs_am[:, None] * a_row_stride + offs_k[None, :] * a_col_stride)
    b_ptrs = b_ptr + (offs_k[:, None] * b_row_stride + offs_bn[None, :] * b_col_stride)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    # k is now an iteration counter (0, 1, 2, ...), not an element offset
    # like old's `range(0, K, BLOCK_SIZE_K)`. Iteration count is identical.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Only the K dimension is masked. K out-of-bounds elements WOULD
        # corrupt valid output rows/cols if accumulated, so we zero them
        # via `other=0.0`. The mask is 1D in K, evaluated against the
        # per-iteration tail size K - k * BLOCK_SIZE_K.
        # Equivalent to old's `k_offsets < K`: that's `(k*BSK + j) < K`,
        # i.e. `j < K - k*BSK`, i.e. `offs_k < K - k*BSK`. Only the LAST
        # iteration ever has any element masked off.
        a_vals = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_vals = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        acc = tl.dot(a_vals, b_vals, acc)

        # Advance the entire pointer block by one K-tile. No 2D recompute.
        # This scalar add reproduces old's per-iter shift: at iter n, old's
        # a_offsets equalled iter-0's plus n*BSK*a_col_stride. Same for B,
        # but along K's row axis (b_row_stride), since K is rows in B.
        a_ptrs += BLOCK_SIZE_K * a_col_stride
        b_ptrs += BLOCK_SIZE_K * b_row_stride

    # Final store uses the REAL M/N bounds, discarding any wrap-around
    # output positions produced by the modulo trick. This is the ONE place
    # the M/N boundary check is paid -- the old kernel paid it every K iter.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * c_row_stride + offs_cn[None, :] * c_col_stride
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matrix_multiplication_pointer(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matrix_multiplication_pointer_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
    )
    return c



@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_persistent_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_start = tl.program_id(0)
    p_count = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_total = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Loop-invariant stride assumes hoisted out of the persistent loop.
    tl.assume(a_row_stride > 0)
    tl.assume(a_col_stride > 0)
    tl.assume(b_row_stride > 0)
    tl.assume(b_col_stride > 0)
    tl.assume(c_row_stride > 0)
    tl.assume(c_col_stride > 0)

    # Persistent loop: each of the p_count launched programs sweeps every
    # `p_count`-th tile of the output. The launcher caps p_count at NUM_SMS,
    # which is the entire point -- otherwise this degenerates to one tile
    # per program and the loop wrapper buys nothing.
    for pid in range(pid_start, num_pid_total, p_count):

        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        local_pid = pid % num_pid_in_group
        pid_m = first_pid_m + (local_pid % group_size_m)
        pid_n = local_pid // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)


        a_ptrs = a_ptr + (offs_am[:, None] * a_row_stride + offs_k[None, :] * a_col_stride)
        b_ptrs = b_ptr + (offs_k[:, None] * b_row_stride + offs_bn[None, :] * b_col_stride)

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
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


def matrix_multiplication_persistent(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    # Cap the grid at NUM_SMS. If the workload is tiny enough that there are
    # fewer output tiles than SMs, drop to that smaller number so we don't
    # launch programs whose persistent loop would execute zero iterations.
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        ),
    )
    matrix_multiplication_persistent_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
    )
    return c


@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_block_pointers_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid_start = tl.program_id(0)
    p_count = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_total = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tl.assume(a_row_stride > 0)
    tl.assume(a_col_stride > 0)
    tl.assume(b_row_stride > 0)
    tl.assume(b_col_stride > 0)
    tl.assume(c_row_stride > 0)
    tl.assume(c_col_stride > 0)

    for pid in range(pid_start, num_pid_total, p_count):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        local_pid = pid % num_pid_in_group
        pid_m = first_pid_m + (local_pid % group_size_m)
        pid_n = local_pid // group_size_m
        
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        
        a_block_ptr = tl.make_block_ptr(
            a_ptr,
            shape=(M, K),
            strides=(a_row_stride, a_col_stride),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1,0),
        )
        b_block_ptr = tl.make_block_ptr(
            b_ptr,
            shape=(K, N),
            strides=(b_row_stride, b_col_stride),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1,0),
        )
        c_block_ptr = tl.make_block_ptr(
            c_ptr,
            shape=(M, N),
            strides=(c_row_stride, c_col_stride),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1,0),
        )
        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a_vals = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
            b_vals = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")

            acc = tl.dot(a_vals, b_vals, acc)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

def matrix_multiplication_block_pointers(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    # Same persistent-grid sizing as matrix_multiplication_persistent: cap at
    # NUM_SMS so each program walks multiple tiles via the persistent loop,
    # and drop below NUM_SMS only when the workload has fewer tiles than SMs.
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        ),
    )
    matrix_multiplication_block_pointers_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
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
c_triton_autotuned = matrix_multiplication_autotuned(a, b)
c_triton_pointer = matrix_multiplication_pointer(a, b)
c_triton_persistent = matrix_multiplication_persistent(a, b)
c_triton_block_pointers = matrix_multiplication_block_pointers(a, b)
c_torch = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2), (c_triton, c_torch)
assert torch.allclose(c_triton_blocked, c_torch, atol=1e-2, rtol=1e-2), (c_triton_blocked, c_torch)
assert torch.allclose(c_triton_row_major, c_torch, atol=1e-2, rtol=1e-2), (c_triton_row_major, c_torch)
assert torch.allclose(c_triton_tiled, c_torch, atol=1e-1, rtol=1e-1), (c_triton_tiled, c_torch)
assert torch.allclose(c_triton_tiled_supergrouped, c_torch, atol=1e-1, rtol=1e-1), (c_triton_tiled_supergrouped, c_torch)
assert torch.allclose(c_triton_autotuned, c_torch, atol=1e-1, rtol=1e-1), (c_triton_autotuned, c_torch)
assert torch.allclose(c_triton_pointer, c_torch, atol=1e-1, rtol=1e-1), (c_triton_pointer, c_torch)
assert torch.allclose(c_triton_persistent, c_torch, atol=1e-1, rtol=1e-1), (c_triton_persistent, c_torch)
assert torch.allclose(c_triton_block_pointers, c_torch, atol=1e-1, rtol=1e-1), (c_triton_block_pointers, c_torch)
# %%
# Benchmark in fp16: plain tiled (2D grid) vs supergrouped (1D grid) vs torch.
# Both Triton kernels use the same block sizes (64x64x64) and Triton's
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
            'triton_autotuned',
            'triton_pointer',
            'triton_persistent',
            'triton_block_pointers',
            'torch',
        ],
        line_names=[
            "Triton Tiled (2D grid)",
            "Triton Supergrouped (1D grid)",
            "Triton Autotuned (supergrouped + autotune)",
            "Triton Pointer (autotuned + ptr-advance + modulo mask)",
            "Triton Persistent (pointer + persistent NUM_SMS-grid)",
            "Triton Block Pointers (make_block_ptr + boundary_check)",
            "Torch (fp16, cuBLAS)",
        ],
        styles=[('orange', '-'), ('blue', '-'), ('red', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('green', '-')],
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
    if provider == 'triton_autotuned':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_autotuned(a, b))
    if provider == 'triton_pointer':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_pointer(a, b))
    if provider == 'triton_persistent':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_persistent(a, b))
    if provider == 'triton_block_pointers':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_block_pointers(a, b))
    # FLOPs for matmul: 2 * M * N * K (one multiply + one add per output element per K dim)
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark.run(show_plots=True, print_data=True)
# %%
