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

If you have fewer bits, you have fewer to move around.