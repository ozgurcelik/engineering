# Tensor Parallelism

Imagine a matrix multiplication of two matrices X and A. Assume that the input matrix is of shape [n, m] and the weights matrix is of shape [m, k].

$$
Y = XA = \begin{bmatrix}
x_1 & x_2 \\
x_3 & x_4
\end{bmatrix} \begin{bmatrix}
a_1 & a_2 \\
a_3 & a_4
\end{bmatrix} = \begin{bmatrix}
x_1a_1 + x_2a_3 & x_1a_2 + x_2a_4 \\
x_3a_1 + x_4a_3 & x_3a_2 + x_4a_4
\end{bmatrix}
$$

We can do the sharding of the weights matrix A in two ways:
1. Row-wise sharding
2. Column-wise sharding

## Row-wise sharding

In row-wise sharding, we split the weights matrix A into two row-wise parts, and shard the input matrix X into two column-wise parts for it to match the row-wise parts of A.
In case of two ranks, each rank will have [n, m/2] data, and [m/2, k] weights matrix.
Each rank will generate [n, k] output matrix, but with partial results.
At the end, we need to sum the partial results to get the final result.

$$
\begin{aligned}
Y = XA &= \left[\begin{array}{c|c}
x_1 & x_2 \\
x_3 & x_4
\end{array}\right] \left[\begin{array}{cc}
a_1 & a_2 \\
\hline
a_3 & a_4
\end{array}\right] = \begin{bmatrix} x_1 \\ x_3 \end{bmatrix} \begin{bmatrix} a_1 & a_2 \end{bmatrix} + \begin{bmatrix} x_2 \\ x_4 \end{bmatrix} \begin{bmatrix} a_3 & a_4 \end{bmatrix} \\
&= X_A A_A + X_B A_B \\
&= \begin{bmatrix} x_1a_1 + x_2a_3 & x_1a_2 + x_2a_4 \\ x_3a_1 + x_4a_3 & x_3a_2 + x_4a_4 \end{bmatrix}
\end{aligned}
$$

Where the subscripts A and B denote the shard of the matrix for the rank A and B respectively.

## Column-wise sharding

In column-wise sharding, we split the weights matrix by columns.
The ranks will have the same, original data matrix X with shape [n, m], but in the case of two ranks, each rank will have [m, k/2] weights matrix. Each rank will generate [n, k/2] output matrix, but with complete results for the parts that it has.
At the end, we need to concatenate the two matrices to get the final result.

$$
\begin{aligned}
Y = XA &= \left[\begin{array}{cc}
x_1 & x_2 \\
x_3 & x_4
\end{array}\right] \left[\begin{array}{c|c}
a_1 & a_2 \\
a_3 & a_4
\end{array}\right]
= \left\lbrack\left[\begin{array}{cc}
x_1 & x_2 \\
x_3 & x_4
\end{array}\right] \begin{bmatrix} a_1 \\ a_3 \end{bmatrix} ,
\left[\begin{array}{cc}
x_1 & x_2 \\
x_3 & x_4
\end{array}\right] \begin{bmatrix} a_2 \\ a_4 \end{bmatrix}
\right\rbrack\ \\
&= (X A_A , X A_B) \\
&= \begin{bmatrix} x_1a_1 + x_2a_3 & x_1a_2 + x_2a_4 \\ x_3a_1 + x_4a_3 & x_3a_2 + x_4a_4 \end{bmatrix}
\end{aligned}
$$

So, for the column-wise sharding, we need to concatenate the two matrices X A_A and X A_B to get the final result, while for the row-wise sharding, we need to sum the two matrices X A_A and X A_B to get the final result.

## Backward pass

Still, we are looking at the 

$$
Y = XA = \begin{bmatrix}
x_1 & x_2 \\
x_3 & x_4
\end{bmatrix} \begin{bmatrix}
a_1 & a_2 \\
a_3 & a_4
\end{bmatrix} = \begin{bmatrix}
x_1a_1 + x_2a_3 & x_1a_2 + x_2a_4 \\
x_3a_1 + x_4a_3 & x_3a_2 + x_4a_4
\end{bmatrix}
$$

For the backward pass, we need to compute the gradients of the loss with respect to the weights matrix A and the input matrix X.

### Row-wise sharding

Say $X_0 = slice(X, 0)$ and $X_1 = slice(X, 1)$, then we have:

**Shapes:** $X: [n, m]$, $X_0, X_1: [n, m/2]$, $A_0, A_1: [m/2, k]$, $F, G, Y: [n, k]$.

$$
\begin{aligned}
Y &= F + G = \underbrace{X_0}_{[n,\, m/2]} \underbrace{A_0}_{[m/2,\, k]} + \underbrace{X_1}_{[n,\, m/2]} \underbrace{A_1}_{[m/2,\, k]} & \quad &[n, k] \\[6pt]
\frac{\partial Y}{\partial X} &= \underbrace{\frac{\partial Y}{\partial F}}_{= I} \underbrace{\frac{\partial F}{\partial X}}_{[nk \times nm]} + \underbrace{\frac{\partial Y}{\partial G}}_{= I} \underbrace{\frac{\partial G}{\partial X}}_{[nk \times nm]} & \quad &[nk \times nm] \\[6pt]
&= \underbrace{\frac{\partial Y}{\partial F}}_{= I\;[nk \times nk]} \underbrace{\frac{\partial F}{\partial X_0}}_{[nk \times \frac{nm}{2}]} \underbrace{\frac{\partial X_0}{\partial X}}_{[\frac{nm}{2} \times nm]} + \underbrace{\frac{\partial Y}{\partial G}}_{= I\;[nk \times nk]}\underbrace{\frac{\partial G}{\partial X_1}}_{[nk \times \frac{nm}{2}]} \underbrace{\frac{\partial X_1}{\partial X}}_{[\frac{nm}{2} \times nm]} & \quad &[nk \times nm] \\[6pt]
&= \underbrace{A_0^T}_{[k,\, m/2]} \underbrace{\frac{\partial X_0}{\partial X}}_{\text{select cols } 0..\frac{m}{2}} + \underbrace{A_1^T}_{[k,\, m/2]} \underbrace{\frac{\partial X_1}{\partial X}}_{\text{select cols } \frac{m}{2}..m} & \quad &[k \times m] \\[6pt]
&= \underbrace{concat(A_0^T, A_1^T)}_{[k,\, m] \;=\; A^T}
\end{aligned}
$$

> **Note on Jacobian dimensions:** Since $Y$, $F$, $G$, $X$ are all matrices, the true Jacobians are 4-tensors. To write them as 2D matrices we vectorize (flatten) each matrix, e.g. $Y: [n,k]$ becomes $\text{vec}(Y): [nk, 1]$, giving Jacobians like $\frac{\partial \text{vec}(Y)}{\partial \text{vec}(X)}: [nk \times nm]$. In the last two lines we drop back to the compact matrix shorthand where $\frac{\partial F}{\partial X_0} = A_0^T$ means the backward pass multiplies by $A_0^T$ on the right.

> **Note on $\frac{\partial X_0}{\partial X}$:** $X_0$ is just the first $m/2$ columns of $X$, so this Jacobian is a column-selector: it routes gradients into the first $m/2$ columns of $\frac{\partial L}{\partial X}$ (zeros elsewhere). Likewise $\frac{\partial X_1}{\partial X}$ routes into the last $m/2$ columns. Since they target non-overlapping columns, the sum of the two terms becomes a column-wise concatenation.

Finally,

$$
\frac{\partial l}{\partial X} = \frac{\partial l}{\partial X} concat(A_0^T, A_1^T)
$$

Since the $A_0$ and $A_1$ are on the different ranks, we can do an all-gather to do the concatenation.

### Column-wise sharding

Shapes: $X: [n, m]$, $A_0, A_1: [m, k/2]$, $F, G, Y: [n, k/2]$.

$$
\begin{aligned}
Y &= F + G = X A_0 + X A_1 \\
\frac{\partial l}{\partial X} &= \frac{\partial l}{\partial Y} (\frac{\partial Y}{\partial F} \frac{\partial F}{\partial X} + \frac{\partial Y}{\partial G} \frac{\partial G}{\partial X}) \\
&= slice(\frac{\partial l}{\partial Y}, 0) A_0^T + slice(\frac{\partial l}{\partial Y}, 1) A_1^T
\end{aligned}
$$

Since the $A_0$ and $A_1$ are on the different ranks, we can do an all-reduce to calculate their sum.

## Implementation details (column-wise sharding)

### Communication pattern summary

| Direction | Operation | Purpose |
|-----------|-----------|---------|
| Forward   | all-gather | Collect local outputs $[n, k/R]$ from each rank into full $[n, k]$ for the next layer |
| Backward  | slice (local) | Reverse of all-gather: each rank takes its $[n, k/R]$ portion of the upstream gradient |
| Backward  | all-reduce (sum) | Sum partial input gradients across ranks to get full $\frac{\partial l}{\partial X}$ |

### Definition of $g_r$

The forward pass on rank $r$ for a single layer computes $Y_r = X A_r$ (matmul) followed by $\text{relu}(Y_r)$ (activation). $g_r$ is the gradient of the loss w.r.t. the matmul output $Y_r$:

$$
g_r = \frac{\partial l}{\partial Y_r} = \frac{\partial l}{\partial \text{relu}(Y_r)} \odot \text{relu}'(Y_r)
$$

where $\frac{\partial l}{\partial \text{relu}(Y_r)}$ is the upstream gradient for rank $r$'s post-relu output and $\odot$ is element-wise multiplication. Since $\text{relu}'(Y_r) = \mathbb{1}[Y_r > 0]$, this zeros out entries where the relu was inactive.

### Why param gradients are local (no communication)

Each rank's weight shard $A_r$ only affects its own local output $Y_r = X A_r$. No other rank's computation depends on $A_r$. Therefore:

$$
\frac{\partial l}{\partial A_r} = X^T \cdot g_r
$$

Both $X$ (the layer input, same on all ranks after all-gather) and $g_r$ (local to this rank) are already available. No communication needed.

### Why input gradients need all-reduce

The layer input $X$ feeds into every rank's matmul. Each rank computes a partial gradient:

$$
\left(\frac{\partial l}{\partial X}\right)_r = g_r \cdot A_r^T \quad [n, m]
$$

This is a **dense** $[n, m]$ matrix (not sparse). The full gradient is the sum of all ranks' contributions:

$$
\frac{\partial l}{\partial X} = \sum_{r=0}^{R-1} g_r \cdot A_r^T
$$

Since each $A_r$ lives on a different rank, we all-reduce (sum) to compute the total.

> **Contrast with row-wise sharding:** In row-wise sharding, each rank's partial input gradient fills a different, non-overlapping slice of columns (zeros elsewhere). The terms don't overlap, so the sum becomes a concatenation, and we use **all-gather** instead of all-reduce.

### The subloss trick (per-layer backward with autograd)

We cannot call `loss.backward()` end-to-end because `all_gather` breaks the autograd graph. Instead, we reconstruct the local computation for each layer and use autograd on that small graph.

For each layer going backward, we have `grad`: the gradient of the loss w.r.t. the gathered output of this layer, shape $[n, k]$.

**Last layer:** The loss $l = \| Y_{\text{gathered}} \|^2$ decomposes over ranks because each rank's output occupies a non-overlapping slice:

$$
l = \sum_{r} \| Y_r \|^2
$$

So `subloss = x.square().sum()` on each rank produces the correct local gradients.

**Intermediate layers:** We use a linear approximation. Given the upstream gradient $g$ for this rank's local output:

$$
\text{subloss} = \sum_{ij} x_{ij} \cdot g_{ij}
$$

Differentiating w.r.t. any variable $v$:

$$
\frac{\partial \text{subloss}}{\partial v} = \sum_{ij} g_{ij} \cdot \frac{\partial x_{ij}}{\partial v}
$$

This is exactly the chain rule: $g$ acts as the upstream gradient flowing into $x$. So `subloss.backward()` produces the same result as `x.backward(gradient=g)`, letting autograd handle relu, matmul, or any other differentiable operation without manual gradient formulas.

### Memory cost of storing activations

The forward pass stores `layer_inputs[i]` (shape $[n, k]$) for each layer. This is the same memory cost as standard autograd, which also retains intermediate activations internally. It could be reduced with **activation checkpointing**: store only every $N$-th layer's input and recompute the rest during backward by replaying the forward from the nearest checkpoint. This trades compute for memory, but we skip it here for clarity.

# Pipeline Parallelism

Pipeline parallelism is a technique where we split the model into multiple stages, and each stage is run on a different device.

Each stage takes the entire input tensor and passes it through its own set of weights.
After that, it send the output to the next stage.

This would mean the device that holds a stage would wait for the previous stage to finish before it can start its computation, and would sit idle until its turn comes again (either in backpropagation or forward pass if inference) after sending the output to the next stage. With $n$ devices, the utilization would be $1/n$ which is very low.

The solution to that is to process micro-batches and send them to the next stage immediately.

## Forward pass

Each rank owns a contiguous group of layers (a "stage"). The full batch is split into $M$ micro-batches. Each micro-batch flows through the pipeline via point-to-point `send`/`recv`:

$$
\text{rank 0} \xrightarrow{\text{send}} \text{rank 1} \xrightarrow{\text{send}} \cdots \xrightarrow{\text{send}} \text{rank } R{-}1
$$

For micro-batch $i$, stage $r$ receives an activation tensor $X_r^{(i)}$ from stage $r-1$ (or uses the original data if $r = 0$), runs it through its local layers, and sends the output to stage $r+1$:

$$
Y_r^{(i)} = f_r(X_r^{(i)}) = \text{relu}(\cdots \text{relu}(X_r^{(i)} W_r^{(1)}) \cdots W_r^{(L_r)})
$$

where $W_r^{(1)}, \ldots, W_r^{(L_r)}$ are the weight matrices for the $L_r$ layers owned by stage $r$.

The output of stage $r$ becomes the input to stage $r+1$: $X_{r+1}^{(i)} = Y_r^{(i)}$.

## Backward pass

The loss is computed only on the last stage ($r = R-1$) as the sum of per-micro-batch losses:

$$
L = \sum_{i=0}^{M-1} L^{(i)} = \sum_{i=0}^{M-1} \| Y_{R-1}^{(i)} \|^2
$$

Gradients flow in reverse through the pipeline via point-to-point communication:

$$
\text{rank } R{-}1 \xrightarrow{\text{send grad}} \text{rank } R{-}2 \xrightarrow{\text{send grad}} \cdots \xrightarrow{\text{send grad}} \text{rank 0}
$$

For each micro-batch $i$, each stage $r$ needs to compute two things:

### 1. Parameter gradients (local, no communication)

Each stage's weights $W_r$ only affect its own local computation. Given the upstream gradient $G_r^{(i)} = \frac{\partial L^{(i)}}{\partial Y_r^{(i)}}$, autograd computes $\frac{\partial L^{(i)}}{\partial W_r}$ by backpropagating through the local layers (matmuls and relus). Since both the stage input $X_r^{(i)}$ and the upstream gradient $G_r^{(i)}$ are available locally, no communication is needed.

Gradients accumulate across micro-batches:

$$
\frac{\partial L}{\partial W_r} = \sum_{i=0}^{M-1} \frac{\partial L^{(i)}}{\partial W_r}
$$

### 2. Input gradient (sent to previous stage)

The input gradient is needed by the previous stage as its upstream gradient:

$$
\frac{\partial L^{(i)}}{\partial X_r^{(i)}} = \frac{\partial L^{(i)}}{\partial Y_r^{(i)}} \cdot \frac{\partial Y_r^{(i)}}{\partial X_r^{(i)}}
$$

This is computed by autograd as `inp.grad` and sent to stage $r-1$ via `dist.send`. Stage $r-1$ receives it via `dist.recv` and uses it as its upstream gradient $G_{r-1}^{(i)}$.

### Communication pattern

| Direction | Operation | Purpose |
|-----------|-----------|---------|
| Forward   | `send`/`recv` (point-to-point) | Pass micro-batch activations to the next stage |
| Backward  | `send`/`recv` (point-to-point) | Pass upstream gradients to the previous stage |

> **Contrast with tensor parallelism:** Tensor parallelism uses *collective* operations (all-gather, all-reduce) because every rank participates in computing each layer. Pipeline parallelism uses *point-to-point* operations because each layer lives entirely on one rank — communication only happens at stage boundaries.

## Implementation details

### The subloss trick (same as tensor parallelism)

We cannot call `loss.backward()` end-to-end because `send`/`recv` breaks the autograd graph (just as `all_gather` does in tensor parallelism). Instead, for each micro-batch we reconstruct the local forward computation and use the subloss trick:

**Last stage ($r = R-1$):** The loss decomposes over micro-batches, so `subloss = x.square().sum()` gives the correct local gradients.

**Other stages:** Given upstream gradient $G_r^{(i)}$ received from stage $r+1$:

$$
\text{subloss} = \sum_{jk} x_{jk} \cdot G_{r,jk}^{(i)}
$$

This is the same linear proxy used in tensor parallelism. `subloss.backward()` produces the same result as `x.backward(gradient=G_r^{(i)})`.

### Gradient accumulation across micro-batches

`optimizer.zero_grad()` is called once per step. Each micro-batch's `subloss.backward()` call *accumulates* into `params[layer].grad`. After all micro-batches are processed, a single `optimizer.step()` applies the total gradient. This is mathematically equivalent to computing the loss over the full batch.

### Synchronization via blocking send/recv

All ranks enter the same `for i in range(num_micro_batches)` backward loop simultaneously. The blocking nature of `send`/`recv` creates the correct execution order without explicit scheduling:

1. Non-last ranks immediately block on `dist.recv(src=rank+1)`
2. The last rank computes its subloss backward and calls `dist.send(dst=rank-1)`
3. Rank $R-2$ unblocks, computes, sends to rank $R-3$, etc.

Within each micro-batch, execution is sequential from last rank to first. There is no cross-micro-batch overlap in this schedule.

### Pipeline bubble

In the F-then-B schedule, each rank is idle while waiting for activations (forward) or gradients (backward) from adjacent stages. With $R$ stages and $M$ micro-batches, the "bubble" — the fraction of time a device is idle — is roughly:

$$
\text{bubble} \approx \frac{R - 1}{R - 1 + M}
$$

Increasing $M$ (more micro-batches) shrinks the bubble but increases memory usage (more stored activations). More advanced schedules like 1F1B (one-forward-one-backward) interleave forward and backward passes to reduce peak memory while maintaining the same bubble ratio.