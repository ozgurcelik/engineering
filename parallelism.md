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