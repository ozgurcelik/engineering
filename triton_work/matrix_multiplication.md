# Matrix Multiplication in Triton

We will now look at matrix multiplication in Triton, which is the core operation in GEMM (General Matrix Multiplication).
Assume we have two matrices A and B, and we want to compute the matrix product C = A @ B.
The matrix A is of size MxK and the matrix B is of size KxN, and the matrix C is of size MxN.

## Naive Implementation

The C[i,j] element is computed as:
```
C[i,j] = sum_{k=0}^{K-1} A[i,k] * B[k,j]
```

The naive implementation is to compute each element of the C separately in a different program.

```python
@triton.jit
def matrix_multiplication_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    # Accumulate in fp32 for both fp16 and fp32 inputs. tl.store casts the
    # result to c_ptr's element type, so the output dtype matches the input.
    acc = tl.zeros([], dtype=tl.float32)
    for k in range(K):
        a_val = tl.load(a_ptr + row * a_row_stride + k * a_col_stride)
        b_val = tl.load(b_ptr + k * b_row_stride + col * b_col_stride)
        acc += a_val.to(tl.float32) * b_val.to(tl.float32)
    c_m_n_ptr = c_ptr + row * c_row_stride + col * c_col_stride
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_kernel_naive[(M, N)](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
    )
    return c
```

We launch an MxN grid, matching the shape of C, and each program computes one output element.
For C[i,j], the program IDs provide `i = tl.program_id(0)` and `j = tl.program_id(1)`.
At inner-loop step k, the program loads A[i,k] and B[k,j].
The address of A[i,k] is `a_ptr + i * a_row_stride + k * a_col_stride`.
Similarly, the address of B[k,j] is `b_ptr + k * b_row_stride + j * b_col_stride`.
Passing both row and column strides allows the kernels to handle strided inputs, including transposed matrices.
The scalar accumulator starts at zero and remains FP32 for both supported input dtypes.
Finally, `tl.store` converts the FP32 accumulator to C's element type, so the output has the same dtype as the inputs.

For each output element, this implementation requests K elements from A and K elements from B, then stores one element in C.
Using the standard GEMM counting convention, it performs approximately 2K FLOPs per output element: K multiplications and K additions.

## Naive Blocked Implementation

The question we want to answer is would doing the inner loop in blocks of size BLOCK_SIZE_K improve the performance?

```python
@triton.jit
def matrix_multiplication_kernel_naive_blocked(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = tl.zeros([], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_start_ptr = a_ptr + row * a_row_stride
        b_start_ptr = b_ptr + col * b_col_stride

        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets * a_col_stride
        b_offsets = k_offsets * b_row_stride

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        acc += tl.sum(a_vals * b_vals)

    c_m_n_ptr = c_ptr + row * c_row_stride + col * c_col_stride
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive_blocked(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_K = 128
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_kernel_naive_blocked[(M, N)](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_K,
    )
    return c
```

This implementation loads the inner loop in blocks of size BLOCK_SIZE_K.
Instead of loading individual A[i,k] and B[k,j] values, it A[i,k_start:k_end] and B[k_start:k_end,j].
Since the K may not be divisible by BLOCK_SIZE_K, we need to use a mask to prevent out-of-bounds loads.

While the total number of FLOPs is the same, there are 2 significant differences between the two implementations.
First, instead of loading, multiplying, and adding 1 element at a time, we are loading BLOCK_SIZE_K elements at a time, multiplying, and adding them.
This reduces the instruction count, and increases the parallelism.
Second, the access to A matrix is coalesced since we are loading contiguous values from the A matrix.
On the other hand, the access to B matrix is not coalesced since we are loading values from different rows of the B matrix.

Let's benchmark these two implementations with FP16 matrices and compare them with the torch matmul function.
We will be using square matrices here.

![FP16 matrix multiplication performance: naive Triton, blocked naive Triton, and PyTorch](figures/matmul_naive_vs_naiveblocked_fp16.png)

We can see that the blocked naive Triton implementation is ~10x faster than the naive Triton implementation while performing way worse than the torch matmul function as one can expect.

## Tiled Implementation

In previous implementation, we moved to a blocked implementation for the inner loop.
Now, we will also use blocks for the outer loop.
In tile matmul, each program computes one output tile C[m_tile, n_tile] of size BLOCK_SIZE_M x BLOCK_SIZE_N.
To compute this, the program reads
- One row strip of A: rows m_tile * BLOCK_SIZE_M to m_tile * BLOCK_SIZE_M + BLOCK_SIZE_M - 1, all K columns. Call this A_m
- One column strip of B: columns n_tile * BLOCK_SIZE_N to n_tile * BLOCK_SIZE_N + BLOCK_SIZE_N - 1, all K rows. Call this B_n

Thanks to the tiling, access to the both A and B matrices is coalesced.

```python
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
        
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a_vals, b_vals, acc)

    c_offsets = m_offsets[:, None] * c_row_stride + n_offsets[None, :] * c_col_stride
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)

def matrix_multiplication_tiled(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_tiled_kernel[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
```

First difference we should realize is that the grid size is now `(triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))` instead of `(M, N)`.
That's because we are now computing one tile of the result matrix at a time in one program.
The row and column indices are also affected by the tiling as we can see, and we need to compute the offset and masks for the rows of A and the columns of B, m and n respectively.
Note that the accumulator is now a 2D array of size BLOCK_SIZE_M x BLOCK_SIZE_N, the same size as the output tile.

As we discussed, to compute the output tile C[m_tile, n_tile], we need to compute the dot product of A_m and B_n, which are now 2D arrays of size BLOCK_SIZE_M x K and K x BLOCK_SIZE_N respectively.
We do this operation in blocks of size BLOCK_SIZE_K in the inner loop.
The offsets for A and B are computed as `m_offsets[:, None] * a_row_stride + k_offsets[None, :] * a_col_stride` and `k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride` respectively.
Quite a bit symmetric as we like to see.
Then we compute the masks for A and B, `m_mask[:, None] & k_mask[None, :]` and `k_mask[:, None] & n_mask[None, :]` respectively.
We should realize that since we have both inner and outer blocks, the masks may have False values in both dimensions.

Finally, we do the same offset and mask computation for the output tile and save the results.

Now, let's compare the performance of the tiled implementation with different block sizes (BLOCK_SIZE_MxBLOCK_SIZE_NxBLOCK_SIZE_K) and compare it with the torch matmul function.
We will be using square matrices once again.

![FP16 matrix multiplication performance: tiled Triton, and PyTorch](figures/matmul_tiled_blocksizes.png)

matmul-tiled-vs-torch-fp16:
          M        N        K  Triton 64x64x64  Triton 128x128x64      Torch
0     256.0    256.0    256.0         3.321312           2.617146   3.063213
1     512.0    512.0    512.0        16.344097          13.393646  17.739589
2    1024.0   1024.0   1024.0        40.089692          32.203279  33.063282
3    2048.0   2048.0   2048.0        52.578786          58.014931  60.604645
4    4096.0   4096.0   4096.0        43.445212          53.805863  52.393391
5    4608.0   4608.0   4608.0        40.038352          48.826759  52.697203
6    5120.0   5120.0   5120.0        26.867392          37.117943  54.394575
7    6144.0   6144.0   6144.0        17.078184          29.612991  64.068548
8    7168.0   7168.0   7168.0        21.833254          28.313260  59.709510
9    8192.0   8192.0   8192.0        17.893498          26.176944  56.509382
10  16384.0  16384.0  16384.0        14.667996          28.089333  55.831171

Remember that for the L4 GPU, we have the following properties:
- 58 SMs
- 48 MB L2
- 300 GB/s memory bandwidth
- 121 TFLOP/s FP16 performance

Now, let's try to make sense of the results we are seeing.

### Why do we have a large drop in performance as we increase the matrix size from 4608 to 5120?

As we said before, we are using FP16 square matrices here.
The memory of such a matrix is 2N^2 bytes.
For the 4608x4608 matrix, this is 40.5 MB, and for the 5120x5120 matrix, this is 50 MB.
The largest square matrix that can fit in the L2 cache is

```
\(n_\text{max}
=
\sqrt{\frac{48\cdot2^{20}}{2}}
\approx5017\)
```

In this implementation, we do not have an explicit grouped tile ordering.
Because of that, while for 4096 and 4608 matrices, one of the operands can reasonably fit in the L2 cache, after that point, programs can sweep too far across one grid dimension before returning to an operand tile which has already been evicted from the L2 cache.
Of course, it's not just one operand taking space in the L2 cache, but after that point, 5017, we can't theoritically fit an operand anymore in the L2 cache.

### Why does 128x128 eventually beat 64x64?

One program that computes one output tile of size TxT from A and B matrices of size MxK and KxN respectively performs T^2K multiplications and T^2K additions, totaling 2T^2K FLOPs.
Ignoring the cache reuse, it reads 2TK + 2KT = 4TK bytes of data (FP16 occupies 2 bytes).
This gives an arithmetic intensity of 2T^2K / 4TK = T/2.

Now looking at 64x64 and 128x128 output tiles, we expect

| Output tile | No-L2-reuse arithmetic intensity | DRAM roof at 300 GB/s |
|---|---:|---:|
| `64×64` | 32 FLOP/byte | 9.6 TFLOP/s |
| `128×128` | 64 FLOP/byte | 19.2 TFLOP/s |

At the largest matrix size we tested, 16384x16384, we have

128×128×64: 28.1 TFLOP/s
64×64×64: 14.7 TFLOP/s

Quite a bit consistent with our expectations with both exceeding the DRAM roof at 300 GB/s thanks to some L2 reuse.

### Why does 64x64 do better than 128x128 at smaller matrix sizes?

For a matrix multiplication between two matrices of size MxK and KxN respectively, with tiling size TxT, we launch a grid of size (triton.cdiv(M, T), triton.cdiv(N, T)).
In our case, as we use square matrices, we have (M, N) = (N, N), and as a result, we get 

P = ceiling(N/T)^2

programs. For small matrices, this means

| Matrix | `64×64` programs | `128×128` programs |
|---:|---:|---:|
| 256 | 16 | 4 |
| 512 | 64 | 16 |
| 1024 | 256 | 64 |
| 2048 | 1024 | 256 |

Since we are using an L4 GPU, we have 58 SMs.
But for the matrix size 256x256, only 4 programs are launched for tile size 128x128.
So, at the lower end, we have a severe underutilization of the SMs, and simply put this problem is more severe for 128x128 than 64x64.
This is why 64x64 does better than 128x128 at smaller matrix sizes.

### Why does performance peak around 2048x2048?

We see that for all tile and inner loop block sizes, performance peaks around 2048x2048 input matrices.
Why is that?
Let's look at the memory requirements for the different matrix sizes.

| Matrix | Memory (MB) |
|---:|---:|
| 1024 | 2 |
| 2048 | 8 |
| 4096 | 32 |
| 8192 | 128 |
| 16384 | 512 |

Given that we have 48 MB of L2 cache, up to and including 2048x2048 matrices, we can fit both operands in the L2 cache.
In fact, as output writes can outcompete the operand reads for L2 cache, it's good that we can fit even the output matrix in the L2 cache along with the operands for the 2048x2048 matrix case.
But for the 4096x4096 matrix, we no longer can fit both operands in the L2 cache.
And as we discussed before, starting from 5120x5120 matrices, we can't even fit one operand in the L2 cache which led to significant drop in performance.

## Supergrouped Implementation

As we have said, in tile matmul, each program computes one output tile C[m_tile, n_tile] of size BLOCK_SIZE_M x BLOCK_SIZE_N.
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
    wave 1  pids 0..3   tiles (m=0,n=0,1,2,3)   load A0 + B0,B1,B2,B3            -> 5 HBM
    wave 2  pids 4..7   tiles (m=0,n=4,5,6,7)   A0 hit, load B4,B5,B6,B7         -> 4 HBM
                                        (B0,B1,B2,B3 evicted to make room)
    wave 3  pids 8..11  tiles (m=1,n=0,1,2,3)   load A1 + B0,B1,B2,B3            -> 5 HBM
                                        (A0, B0,B1,B2,B3 evicted to make room)
    wave 4  pids 12..15 tiles (m=1,n=4,5,6,7)   A1 hit, load B4,B5,B6,B7         -> 4 HBM
                                        (B0,B1,B2,B3 evicted to make room)
    ... pattern repeats for m=2,3,4,5

SUPERGROUPED (GROUP_SIZE_M=2):
    wave 1  pids 0..3   tiles (m=0,1,n=0,1)   load A0,A1 + B0,B1          -> 4 HBM
    wave 2  pids 4..7   tiles (m=0,1,n=2,3)   A0,A1 hit, load B2,B3       -> 2 HBM
    wave 3  pids 8..11  tiles (m=0,1,n=4,5)   A0,A1 hit, load B4,B5       -> 2 HBM
                                        (B0,B1 evicted to make room)
    wave 4  pids 12..15 tiles (m=0,1,n=6,7)   A0,A1 hit, load B6,B7       -> 2 HBM
                                        (B2,B3 evicted to make room)
    ... when a group is done, we launch the next group and the pattern repeats

As we can see, the supergrouped mapping is more efficient than the row-major mapping.
Now, let's look at the implementation.

```python
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


def matrix_multiplication_tiled_supergrouped(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 64,
    block_size_n: int = 64,
    block_size_k: int = 64,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n
    BLOCK_SIZE_K = block_size_k
    GROUP_SIZE_M = group_size_m
    # 1D launch grid is required for the supergrouped pid -> (pid_m, pid_n) mapping
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_tiled_kernel_supergrouped[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
```

First of all, notice that we no longer have a 2D grid.
Instead, we have a 1D grid of size `(triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)`.
This is because we will do the mapping ourselves instead of relying on the 2D grid which gives us a row-major mapping.
In the triton kernel, we identify the program id we have and determine the dimensions of the supergrouped grid we will construct.
Since we define the groups as the horizontal strips of programs, they will cover the entire column space of the output matrix C.
So, the `GROUP_SIZE_M` is the height of the group in the vertical direction.
Given the strip size in horizontal direction, we can determine the number of programs in each group as `GROUP_SIZE_M * triton.cdiv(N, BLOCK_SIZE_N)`.

Afterwards, we will need to find our position in the supergrouped grid.
Note that we have `pid = tl.program_id(0)` which is the program id we have from the 1D grid.
But, we need to map it to a 2D position in the supergrouped grid.
We start by identifying the group we are in with `group_id = pid // num_pid_in_group`.
Then, we find the starting position of our group in the vertical direction in the 2D supergrouped grid with `first_pid_m = group_id * GROUP_SIZE_M`.
Note that this is not a program id, but an index in the vertical direction.
We then find the height of the group we are in with `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)`.
This can be less than `GROUP_SIZE_M` if the height of the supergrouped grid is not a multiple of `GROUP_SIZE_M`.

Now, we need to determine the position of our program in the supergrouped grid.
The `local_pid = pid % num_pid_in_group` gives us an absolute index from 0 to `num_pid_in_group - 1`.
Now, we need to determine the position of our program in the group.
This program will be in the `local_pid % group_size_m` row and `local_pid // group_size_m` column.
The column index within the group directly gives us the column index in 2D supergrouped grid since a group covers the entire column space of the output matrix C.
For the row index, we need to add the starting position of the group so we get `pid_m = first_pid_m + (local_pid % group_size_m)` for the row index in the 2D supergrouped grid.

The rest of the computation is identical to the tiled implementation.
Except, instead of using `tl.program_id(0)` and `tl.program_id(1)` to get the row and column indices of the output tile, we use `pid_m` and `pid_n` which are the row and column indices of our program in the supergrouped grid.