# Tensor Parallelism

Imagine a matrix multiplication of two matrices X and A. Assume that the input matrix is of shape [n, m] and the weights matrix is of shape [m, k].

$$
XA = \begin{bmatrix}
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
XA &= \left[\begin{array}{c|c}
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
XA &= \left[\begin{array}{cc}
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