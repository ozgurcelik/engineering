# GPUs

How is a GPU different from a CPU?
CPUs optimize for a few fast threads while GPUs optimize for many threads.
A thread is the smallest unit of execution on a GPU. Each thread runs the **same instructions** but on **different data** — this is the SIMT (Single Instruction, Multiple Threads) model.

GPUs have many more compute units and much less support for branching (control, cache).

CPUs optimize for latency (each thread finishes quickly) while GPUs optimize for throughput (total processed data per unit time).

GPUs have many SMs (streaming multiprocessors) that independently execute blocks (jobs).
Each SM contains many SPs (streaming processors) that execute threads in parallel.

The closer to a memory to the SM, the faster the access is.
L1 and shared memory are inside the SM.
L2 cache is on the die, and global memory chips next to the GPU.

## Execution model of a GPU

Threads: Threads do the work in parallel. All threads execute the same instructions but on different data (SIMT).

Blocks: Groups of threads. Each block runs on a single SM with its own shared memory.

Warp: Threads always execute in groups of 32 called a **warp**. Threads in a warp are contiguous in memory.

So, blocks are assigned to SMs, and each block is divided into warps. Each warp contains 32 threads.

Each thread has its own registers.
Each thread can access its own registers and shared memory within the block.
The information that goes across blocks need to be read/written to global memory which is slow.

## Roofline Model

There are two regimes of performance:
- The memory-bound regime: the GPU is bounded by memory bandwidth, how fast can it read/write data.
- The compute-bound regime: the GPU is utilizing its compute units to the fullest.

In the memory-bound regime, the throughput increases as we increase the operational intensity.
While in the compute-bound regime, the throughput does not increase as we increase the operational intensity.

We want to be on the compute-bound regime where we are utilizing our compute units to the fullest.

## How Do We Make a GPU Fast?

Additional source: https://www.thonking.ai/p/what-shapes-do-matrix-multiplications

There are 6 main techniques to make a GPU fast:
- Control divergence (not a memory bottleneck)
- Low precision computation
- Operator fusion
- Recomputation
- Coalescing memory
- Tiling

While the control divergence is not memory based, the other 5 are.

### Control divergence

GPUs are optimized for SIMT (Single Instruction, Multiple Threads) execution.
So every thread in a warp executes the same instruction at the same time.
Conditionals are fine, but if we do something like:
```
if (thread_id <= 3) {
    A;
} else {
    B;
}
```
then when we do $A$, we will have 4 threads executing $A$ and the rest will be idle.
And when we do $B$, we will the initial 4 threads will be idle while the rest will execute $B$.
This is called control divergence.

### Low precision computation

Arithmetic intensity: #FLOPs / #bytes moved.

#### Bits and bytes

A **bit** is the smallest unit of data — a single 0 or 1. A **byte** is 8 bits grouped together. The relationship is always: **1 byte = 8 bits**.

The number in a data type's name tells you how many **bits** it uses:
- **float32** (FP32): 32 bits = 32 / 8 = **4 bytes** per number
- **float16** (FP16): 16 bits = 16 / 8 = **2 bytes** per number
- **bfloat16** (BF16): 16 bits = **2 bytes** per number (different exponent/mantissa split than FP16)
- **int8**: 8 bits = **1 byte** per number

Why does this matter for GPUs? Every number that a thread reads from or writes to memory costs bytes of bandwidth. A float32 value costs 4 bytes per read/write, while a float16 value costs only 2. So switching from float32 to float16 **halves your memory traffic** for the same operation, which directly helps in the memory-bound regime.

Example from the lecture — elementwise ReLU (\(x = \max(0, x)\)) on a vector of size \(n\):
- **Float32**: 1 read + 1 write = 8 bytes moved per element, 1 FLOP → 1/8 FLOP/byte
- **Float16**: 1 read + 1 write = 4 bytes moved per element, 1 FLOP → 1/4 FLOP/byte

Half the bytes means double the arithmetic intensity, pushing the operation closer to the compute-bound regime. Tensor cores (introduced in Volta/Turing) exploit this further — they perform matrix multiplications in low/mixed precision (e.g., FP16 inputs, FP32 accumulation), making matmuls >10x faster than standard floating point ops.

#### FP16 vs BF16

Both are 16-bit (2 bytes), but they split those 16 bits differently. A floating-point number is stored as three fields: **sign** (positive/negative), **exponent** (the scale/range), and **mantissa** (the precision/significant digits).

- **FP16**: 1 sign + 5 exponent + 10 mantissa — more precision, smaller range (max ~65,504)
- **BF16**: 1 sign + 8 exponent + 7 mantissa — less precision, much larger range (max ~3.4 × 10³⁸, same as FP32)

BF16 keeps the same 8 exponent bits as FP32, so it can represent the same range of magnitudes. This matters for training because gradients and activations can span a huge dynamic range. FP16's narrow range causes values to overflow or underflow more easily, which is why FP16 training often requires loss scaling. BF16 avoids this — you can typically drop it in as a replacement for FP32 without any scaling tricks, at the cost of slightly less precision (7 vs 10 mantissa bits). In practice this precision loss rarely affects model quality, which is why BF16 has become the default for LLM training.

### Operator fusion

If we need to do multiple operations in a row, we can fuse them together to reduce the number of memory reads and writes.

### Recomputation

The idea is doing more compute instead of storing the intermediate results in memory.
For example, in backward pass, we can store the activations and compute the jacobians.
Instead, we can recompute the activations and jacobians in the backward pass.

### Coalescing memory

DRAM (global memory) is read in burst mode.
Each address space is partitioned into burst sections.
Whenever a location is accessed, the entire burst section that contains the location is read into the cache.

Memory accesses are coalesced if all the threads in a warp fall into the same burst section.
Only one DRAM request is made for the entire burst section.

#### Row-major layout

A 2D matrix is stored in memory as a flat 1D array. In **row-major** order (the default in C/CUDA), rows are stored one after another:
```
Matrix:          Memory (flat):
| 1  2  3 |      [1, 2, 3, 4, 5, 6, 7, 8, 9]
| 4  5  6 |       ^row 0^  ^row 1^  ^row 2^
| 7  8  9 |
```
Elements in the same row are adjacent in memory. Elements in the same column are separated by the row width.

#### Coalescing for matrix multiplication

Coalescing is about what all 32 threads in a warp access **simultaneously**, not what a single thread does over time.

Consider \(C = A \times B\), where each thread computes one element of \(C\). Each element \(C[i][j]\) is the dot product of row \(i\) of \(A\) and column \(j\) of \(B\), computed over steps \(k = 0, 1, 2, \ldots\). At each step, every thread reads one element from \(A\) and one from \(B\). Whether those reads are coalesced depends on how threads are assigned.

**Bad: threads along a column of C** (thread 0 does `C[0][0]`, thread 1 does `C[1][0]`, etc.):
- **A**: each thread reads from a different row — `A[0][k]`, `A[1][k]`, `A[2][k]`, ... These addresses are each \(N\) apart, scattered across memory. **Not coalesced.**
- **B**: all threads compute the same column, so they all read `B[k][0]` — the exact same address. This is a **broadcast** (one read serves all threads, fine).

**Good: threads along a row of C** (thread 0 does `C[0][0]`, thread 1 does `C[0][1]`, etc.):
- **A**: all threads compute the same row, so they all read `A[0][k]` — a **broadcast** (fine).
- **B**: each thread reads an adjacent column — `B[k][0]`, `B[k][1]`, `B[k][2]`, ... These addresses are contiguous in row-major memory. One DRAM burst serves the whole warp. **Coalesced.**

In both cases one matrix is broadcast (all threads read the same address) and the other is read by all 32 threads at different addresses. The question is whether those 32 addresses are contiguous (coalesced) or strided (not coalesced).

### Tiling

Idea of grouping and ordering threads to minimize the number of global memory accesses.
Cut the matrix into smaller tiles and load them into the shared memory.

Non-tiled matrix multiplication: each input is read $N$ times from global memory.

Tiled matrix multiplication: each input is read $\frac{N}{T}$ times from global memory, and $T$ times from shared memory within each tile. This is a factor of $T$ reduction in global memory accesses.

#### Complexities of tiling

Tile sizes may not divide the matrix size and lead to low utilization.

Loading tiles are fast if burst align with the matrix.

### Wave Quantization

Imagine a matrix of size 1792x1792. Using tile sizes of 256x128, we get

$$
\frac{1792}{256} \times \frac{1792}{128} = 7 \times 14 = 98
$$

tiles.

But if the matrix is 1793x1793, we get

$$
8 \times 15 = 120
$$

tiles.

An A100 has 108 SMs, so we can handle 108 tiles in one go. With 120 tiles, we need to do another cycle where tiles are very sparse to begin with. This is quite inefficient.