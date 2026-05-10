# %%
import triton
import triton.language as tl
import torch
from einops import rearrange

# %%
@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, # Input pointers
    output_ptr, # Output pointer
    x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim, # Likely 1
    output_stride_row, # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Tile shapes must be known at compile time
):

    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    # (1,0) means row major. meaning for a 2D MxN matrix, the strides are (N,1)
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # boundary_check=(0, 1) means both row and column might be out of bounds
        # padding_option="zero" means that out of bounds elements are padded with 0
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # [ROWS_TILE_SIZE, D_TILE_SIZE]
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # [D_TILE_SIZE]

        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))

"""
Backwards pass
==============

Forward op (a matrix-vector product / "weighted sum along the embedding dim"):

    f : R^{n x d} x R^d  ->  R^n
    f(x, w) = x w
    f(x, w)_i = sum_{j=1..d} x_{i,j} * w_j         for i = 1..n

with
    x in R^{n x d},   w in R^d,   f(x, w) in R^n.

Let
    g := grad_{f(x,w)} L  in  R^n
be the upstream gradient (the gradient of the scalar loss L with respect
to the output of this layer). The autograd engine hands us `g`; our job
is to produce grad_x L and grad_w L.

----------------------------------------------------------------------
Gradient w.r.t. x   (shape n x d, same as x itself)
----------------------------------------------------------------------
Each output coordinate depends only on the corresponding row of x:

    df_i / dx_{i,j} = w_j        and        df_i / dx_{k,j} = 0  for k != i.

So the multivariate chain rule collapses to a single term:

    dL/dx_{i,j} = sum_k (dL/df_k) * (df_k / dx_{i,j})
                = (dL/df_i) * w_j
                = g_i * w_j

In matrix form this is an outer product of g (column) and w (row):

    grad_x L  =  g  w^T          # shape (n,) outer (d,) -> (n, d)

Equivalently, broadcasting:  g[:, None] * w[None, :].

Intuition: each row i of x contributed `<x_i, w>` to f_i. Scaling
that contribution by g_i and differentiating w.r.t. x_{i,j} just
leaves the coefficient of x_{i,j}, which is w_j.

----------------------------------------------------------------------
Gradient w.r.t. w   (shape d, same as w itself)
----------------------------------------------------------------------
Every output coordinate depends on every weight:

    df_i / dw_j = x_{i,j}.

Chain rule sums over the n outputs:

    dL/dw_j = sum_i (dL/df_i) * (df_i / dw_j)
            = sum_i g_i * x_{i,j}

This is exactly the j-th component of  x^T g:

    grad_w L  =  x^T g           # (d, n) @ (n,) -> (d,)

Intuition: w_j was multiplied by column j of x and the resulting
n-vector was added (after scaling by g) into L. So the sensitivity of
L to w_j is the dot product of g with that column.

----------------------------------------------------------------------
Final formulas
----------------------------------------------------------------------
    grad_x L  =  g  w^T          (= grad_{f} L  w^T)
    grad_w L  =  x^T g           (= x^T  grad_{f} L)

Implementation notes for the Triton backward kernels:
  * grad_x is fully parallel across (i, j) -- it's a pure elementwise
    outer product, no reduction. One kernel can tile over rows of x
    and cols of w, loading g once per row-tile and w once per col-tile.
  * grad_w requires a reduction across the n rows. Each program
    instance can accumulate a partial sum over a row-tile and either
    (a) atomic-add into a single grad_w buffer, or (b) write per-tile
    partials and reduce in a second pass. (b) is usually preferred for
    determinism and throughput on large n.

----------------------------------------------------------------------
What the backward kernel actually stores: a walkthrough
----------------------------------------------------------------------
Each program instance "owns" one row-tile of size ROWS_TILE_SIZE. With
N rows in x and a launch of cdiv(N, ROWS_TILE_SIZE) programs, program k
processes rows [k*ROWS_TILE_SIZE, (k+1)*ROWS_TILE_SIZE). Inside that
program only g (= grad_output) for those rows is held constant; w, x,
grad_x, and partial_grad_weight are walked along the D axis as the
loop iterates over D-tiles.

(1) grad_x_block_ptr: a (ROWS_TILE_SIZE, D_TILE_SIZE) tile written ONCE.

    Let i0 = k * ROWS_TILE_SIZE and c0 = i * D_TILE_SIZE (loop var).
    For r in [0, ROWS_TILE_SIZE), c in [0, D_TILE_SIZE):

        grad_x_row[r, c] = g[i0 + r] * w[c0 + c]

    This is exactly dL/dx_{i,j} from the chain rule, with no sum to
    perform. Because programs partition the row axis disjointly and
    each loop iteration covers a different D-tile, every (i, j) cell
    of grad_x is touched by exactly one program in exactly one
    iteration. So we tl.store the result directly -- no atomics, no
    accumulation, no second pass.

(2) partial_grad_weight_block_ptr: a (1, D_TILE_SIZE) per-tile PARTIAL.

    The math wants a sum over all N rows: dL/dw_j = sum_i g_i*x_{i,j}.
    A single program only sees ROWS_TILE_SIZE rows, so it can only
    compute its slice of that sum:

        partial_grad_weight[k, j] = sum_{i in tile k} g_i * x_{i,j}

    Concretely the line

        grad_weight_row = tl.sum(row * grad_output[:, None],
                                 axis=0, keep_dims=True)

    multiplies the x-tile elementwise by g (broadcast down the row
    axis) and reduces along that axis, producing shape (1, D_TILE_SIZE):

        grad_weight_row[0, c] = sum_{r=0..ROWS_TILE_SIZE-1}
                                    x[i0+r, c0+c] * g[i0+r]

    Each program writes into row k of partial_grad_weight, so distinct
    programs target disjoint memory -- still no atomics. The full
    grad_w is then assembled outside the kernel by

        grad_weight = partial_grad_weight.sum(axis=0)        # (D,)

    which performs the cross-tile reduction the kernel deliberately
    avoided.

Why the asymmetry?

    grad_x has shape (N, D); partitioning programs along N gives each
    one disjoint output rows -> direct write, no coordination.

    grad_w has shape (D,); EVERY program contributes to EVERY entry,
    so we must either atomic_add into one buffer or stage per-program
    partials. We choose partials: deterministic (atomic float adds are
    not bit-exact across runs because the summation order varies) and
    usually faster (no contention). Cost: an extra (n_row_tiles, D)
    scratch buffer plus a small follow-up reduction.

Two easy-to-miss mechanics:

  * keep_dims=True on the tl.sum: without it the result is shape
    (D_TILE_SIZE,), but partial_grad_weight_block_ptr has block_shape
    (1, D_TILE_SIZE). tl.store requires shapes to match exactly, so we
    keep the singleton row axis.

  * boundary_check=(1,) on the partial_grad_weight store: dim 0 is
    indexed by row_tile_idx < n_row_tiles by construction, so it can
    never be out of bounds. Only D can be ragged when D is not a
    multiple of D_TILE_SIZE, hence checking dim 1 only.
"""

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr, # Input
    grad_output_ptr, # Grad input
    grad_x_ptr, partial_grad_weight_ptr, # Grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):

    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1,0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE,)

        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :] # (ROWS_TILE_SIZE, D_TILE_SIZE)
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # Never out of bounds for dim 0

        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

# %%
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE # These don't have to be the same
        n_rows, D = x.shape

        # grad_out arrives in the original (non-flattened) output shape; the
        # kernel expects a 1D buffer of length n_rows matching the saved x.
        grad_out = grad_out.contiguous().view(-1)

        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype
        )
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x.view(ctx.input_shape), grad_weight


# %%
# Correctness check: compare the Triton kernel against a pure-PyTorch reference.
if __name__ == "__main__":
    torch.manual_seed(0)

    rows, D = 67, 1024  # non-power-of-2 rows to exercise boundary checks
    x = torch.randn(rows, D, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    y_triton = WeightedSumFunc.apply(x, weight)
    y_torch = (x.detach() * weight.detach()).sum(dim=-1)

    assert y_triton.shape == y_torch.shape, f"shape mismatch: {y_triton.shape} vs {y_torch.shape}"
    max_abs_err = (y_triton - y_torch).abs().max().item()
    max_rel_err = ((y_triton - y_torch).abs() / y_torch.abs().clamp(min=1e-6)).max().item()
    print(f"forward  max abs err: {max_abs_err:.3e}  max rel err: {max_rel_err:.3e}")

    torch.testing.assert_close(y_triton, y_torch, atol=1e-3, rtol=1e-3)
    print("OK: Triton weighted sum matches torch reference.")

    # Backward: compare gradients against a pure-PyTorch reference graph.
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    y_ref = (x_ref * w_ref).sum(dim=-1)

    g = torch.randn_like(y_triton)
    (gx_triton, gw_triton) = torch.autograd.grad(y_triton, (x, weight), grad_outputs=g)
    (gx_torch, gw_torch) = torch.autograd.grad(y_ref, (x_ref, w_ref), grad_outputs=g)

    gx_abs = (gx_triton - gx_torch).abs().max().item()
    gw_abs = (gw_triton - gw_torch).abs().max().item()
    print(f"grad_x   max abs err: {gx_abs:.3e}")
    print(f"grad_w   max abs err: {gw_abs:.3e}")

    torch.testing.assert_close(gx_triton, gx_torch, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(gw_triton, gw_torch, atol=1e-3, rtol=1e-3)
    print("OK: Triton weighted sum backward matches torch reference.")

# %%
@triton.jit
def matrix_multiply_fwd(
    a_ptr, b_ptr,
    output_ptr,
    a_stride_row, a_stride_col,
    b_stride_row, b_stride_col,
    output_stride_row, output_stride_col,
    M, N, K,
    M_TILE_SIZE: tl.constexpr, N_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    row_program_count = tl.cdiv(M, M_TILE_SIZE)
    col_program_count = tl.cdiv(N, N_TILE_SIZE)
    total_program_count = row_program_count * col_program_count
    total_program_count_in_group = GROUP_SIZE_M * col_program_count

    local_group_id = pid // total_program_count_in_group    
    local_group_start_row = local_group_id * GROUP_SIZE_M
    local_group_size = min(GROUP_SIZE_M, row_program_count - local_group_start_row)

    local_pid = pid % total_program_count_in_group
    local_pid_m = local_group_start_row + (local_pid % local_group_size)
    local_pid_n = (local_pid // local_group_size)

    a_block_ptr = tl.make_block_ptr(
        a_ptr,
        shape=(M, K),
        strides=(a_stride_row, a_stride_col),
        offsets=(local_pid_m * M_TILE_SIZE, 0),
        block_shape=(M_TILE_SIZE, K_TILE_SIZE),
        order=(1,0),
    )

    b_block_ptr = tl.make_block_ptr(
        b_ptr,
        shape=(K, N),
        strides=(b_stride_row, b_stride_col),
        offsets=(0, local_pid_n * N_TILE_SIZE),
        block_shape=(K_TILE_SIZE, N_TILE_SIZE),
        order=(1,0),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(M, N),
        strides=(output_stride_row, output_stride_col),
        offsets=(local_pid_m * M_TILE_SIZE, local_pid_n * N_TILE_SIZE),
        block_shape=(M_TILE_SIZE, N_TILE_SIZE),
        order=(1,0),
    )

    acc = tl.zeros((M_TILE_SIZE, N_TILE_SIZE), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, K_TILE_SIZE)):
        a_vals = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_vals = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")

        acc = tl.dot(a_vals, b_vals, acc)

        a_block_ptr = tl.advance(a_block_ptr, (0, K_TILE_SIZE))
        b_block_ptr = tl.advance(b_block_ptr, (K_TILE_SIZE, 0))

    tl.store(output_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

"""
Backwards pass for matrix-matrix multiply
=========================================

Forward op:

    f : R^{m x d} x R^{d x n}  ->  R^{m x n}
    f(A, B) = A B = C
    c_{ij} = sum_{k=1..d} a_{ik} b_{kj}

Upstream gradient handed to us by autograd:

    g_{ij} := dL / dc_{ij},      so  G in R^{m x n}  has the same shape as C.

Goal: compute  grad_A L  in R^{m x d}  and  grad_B L  in R^{d x n}.

----------------------------------------------------------------------
Gradient w.r.t. A   (shape m x d, same as A)
----------------------------------------------------------------------
L depends on a_{ij} only through C, so the chain rule sums over every
entry of C:

    dL / da_{ij} = sum_{k=1..m} sum_{l=1..n} g_{kl} * (dc_{kl} / da_{ij})

To find dc_{kl} / da_{ij}, write c_{kl} explicitly and differentiate:

    c_{kl} = sum_{p=1..d} a_{kp} b_{pl}

    dc_{kl} / da_{ij} = sum_p (da_{kp} / da_{ij}) * b_{pl}
                     = sum_p delta_{ki} delta_{pj} * b_{pl}
                     = delta_{ki} * b_{jl}

Two things to notice:
  * k is pinned to i (delta_{ki}). Row i of A only contributes to row i
    of C, so the outer sum over k collapses.
  * l is NOT pinned. a_{ij} appears in every column l of row i of C
    (with coefficient b_{jl}), so a sum over l survives.

Plugging back in:

    dL / da_{ij} = sum_l g_{il} * b_{jl}

----------------------------------------------------------------------
Gradient w.r.t. B   (shape d x n, same as B)
----------------------------------------------------------------------
Same recipe, differentiating w.r.t. b_{ij} this time:

    dc_{kl} / db_{ij} = sum_p a_{kp} * (db_{pl} / db_{ij})
                     = sum_p a_{kp} * delta_{pi} delta_{lj}
                     = a_{ki} * delta_{lj}

Now l is pinned (to j) and k stays free, exactly mirroring the A case:

    dL / db_{ij} = sum_k g_{kj} * a_{ki}

The asymmetry between the two pinned indices (k pinned for grad_A,
l pinned for grad_B) is exactly why grad_A reduces over n while
grad_B reduces over m.

----------------------------------------------------------------------
Turning indexed sums into matrix products
----------------------------------------------------------------------
The matrix product convention is

    (X Y)_{ij} = sum_l X_{il} Y_{lj}

i.e., the SUMMED index sits in the second slot of the left factor and
the first slot of the right factor; free index i is far left, free
index j is far right. To rewrite an indexed sum as a matrix product,
transpose any factor whose summed index is in the wrong slot.

Case A:  dL / da_{ij} = sum_l g_{il} * b_{jl}
  * g_{il}: summed index l is in slot 2  -> already correct.
  * b_{jl}: summed index l is in slot 2, but as the right factor we
            need l in slot 1. Transpose: b_{jl} = (B^T)_{lj}.
  * Result: sum_l g_{il} (B^T)_{lj} = (G B^T)_{ij}.

Case B:  dL / db_{ij} = sum_k g_{kj} * a_{ki}
  Reorder for clarity:  sum_k a_{ki} * g_{kj}.
  * a_{ki}: summed index k is in slot 1, but as the left factor we
            need k in slot 2. Transpose: a_{ki} = (A^T)_{ik}.
  * g_{kj}: summed index k is in slot 1 -> already correct.
  * Result: sum_k (A^T)_{ik} g_{kj} = (A^T G)_{ij}.

----------------------------------------------------------------------
Final formulas
----------------------------------------------------------------------
    grad_A L  =  G B^T          # (m, n) @ (n, d) -> (m, d)
    grad_B L  =  A^T G          # (d, m) @ (m, n) -> (d, n)

Shape-based shortcut: once you know it must be a product of two of
{G, A, B} (with optional transpose) producing the right shape, there
is only one such combination for each gradient. This is how most
people remember the formulas in practice -- but the index derivation
above is what justifies it.

----------------------------------------------------------------------
Implementation notes for the Triton kernels
----------------------------------------------------------------------
Both backwards are themselves matmuls, so the tl.dot skeleton from
matrix_multiply_fwd carries over almost unchanged. Two important
asymmetries to plan around:

  * grad_A = G B^T  has output shape (m, d) and reduces over n.
    Tile (m, d), inner loop over n.

  * grad_B = A^T G  has output shape (d, n) and reduces over m.
    Tile (d, n), inner loop over m.

Because the reduction axes differ, the optimal tile shapes and grid
launches differ, and production code typically uses TWO separate
kernels rather than trying to compute both in one launch. Each
program writes a disjoint output tile, so neither needs atomics or
partial buffers (unlike the grad_w case in weighted_sum_backward,
where every program contributed to every entry of grad_w).
"""

@triton.jit
def matrix_multiply_backward(
    a_ptr, b_ptr,
    grad_output_ptr,
    

# %%
class MatrixMultiplyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        M, K = a.shape
        K, N = b.shape

        ctx.save_for_backward(a, b)

        assert len(a.shape) == 2 and len(b.shape) == 2, "Expected 2D tensors"
        assert a.shape[1] == b.shape[0], "Dimension mismatch"
        assert a.is_cuda and b.is_cuda, "Expected CUDA tensors"
        assert a.is_contiguous() and b.is_contiguous(), "Expected contiguous tensors"

        ctx.M_TILE_SIZE = 16
        ctx.N_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.GROUP_SIZE_M = 2

        ctx.M = M
        ctx.N = N
        ctx.K = K

        output = torch.empty((M, N), device=a.device, dtype=a.dtype)

        grid = triton.cdiv(M, ctx.M_TILE_SIZE) * triton.cdiv(N, ctx.N_TILE_SIZE)
        matrix_multiply_fwd[(grid,)](
            a, b,
            output,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            output.stride(0), output.stride(1),
            M=M, N=N, K=K,
            M_TILE_SIZE=ctx.M_TILE_SIZE, N_TILE_SIZE=ctx.N_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE,
            GROUP_SIZE_M=ctx.GROUP_SIZE_M,
        )

        return output


# %%
# Correctness check for the MatrixMultiplyFunc forward pass.
# The kernel uses tl.dot (which prefers fp16/bf16 inputs) and casts the
# fp32 accumulator down to fp16 on store, so we test against fp16 inputs.
if __name__ == "__main__":
    torch.manual_seed(0)

    shapes = [
        (64, 64, 64),       # all dims aligned to tile sizes (16)
        (128, 256, 64),     # rectangular, all aligned
        (67, 129, 73),      # all dims unaligned -> exercises boundary checks
        (16, 16, 16),       # single tile
    ]

    for M, K, N in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        out_triton = MatrixMultiplyFunc.apply(a, b)
        out_torch = (a.float() @ b.float()).to(torch.float16)

        assert out_triton.shape == out_torch.shape, (
            f"shape mismatch for ({M},{K})x({K},{N}): "
            f"{out_triton.shape} vs {out_torch.shape}"
        )
        assert out_triton.dtype == torch.float16, f"dtype mismatch: {out_triton.dtype}"

        max_abs_err = (out_triton.float() - out_torch.float()).abs().max().item()
        denom = out_torch.float().abs().clamp(min=1e-3)
        max_rel_err = ((out_triton.float() - out_torch.float()).abs() / denom).max().item()
        print(
            f"matmul ({M:>4},{K:>4})x({K:>4},{N:>4})  "
            f"max abs err: {max_abs_err:.3e}  max rel err: {max_rel_err:.3e}"
        )

        # fp16 accumulation tolerance scales with K (more terms summed).
        atol = 1e-2 * max(1.0, K / 64)
        rtol = 1e-2
        torch.testing.assert_close(
            out_triton.float(), out_torch.float(), atol=atol, rtol=rtol
        )

    print("OK: MatrixMultiplyFunc forward matches torch reference.")

# %%