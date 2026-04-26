# %%
# Playground for Triton-style indexing / broadcasting / masking.
#
# Triton kernels themselves only run on GPU inside @triton.jit, and you can't
# easily inspect intermediate tensors. But the *math* of Triton's pointer
# arithmetic is identical to PyTorch broadcasting:
#
#   tl.arange(0, N)       <->  torch.arange(N)
#   x[:, None], x[None,:] <->  same in PyTorch
#   tl.load(ptr + offs)   <->  flat_tensor[offs]
#   tl.store              <->  flat_tensor[offs] = val
#   tl.sum(x, axis=0)     <->  x.sum(dim=0)
#   masks                 <->  same idea: zero out / skip elements
#
# So we can prototype kernel logic in pure PyTorch first, then port it.
# At the bottom of the file there's also a tiny real Triton kernel using
# tl.device_print so you can sanity-check shapes on the GPU.

import torch

# %%
# 1) tl.arange equivalent
M, N = 4, 6
rows = torch.arange(M)
cols = torch.arange(N)
print("rows:", rows, rows.shape)
print("cols:", cols, cols.shape)

# %%
# 2) The core Triton broadcasting trick: [:, None] and [None, :]
#    Turn two 1-D vectors into a 2-D grid.
row_grid = rows[:, None]                     # shape [M, 1]
col_grid = cols[None, :]                     # shape [1, N]
print("row_grid shape:", row_grid.shape)
print("col_grid shape:", col_grid.shape)

grid = row_grid * 100 + col_grid             # shape [M, N]
print("grid:\n", grid)
print("grid shape:", grid.shape)

# %%
# 3) Reproducing the b_offsets pattern from your row-major matmul.
#    For each k in [0, BK) and each n in [0, BN), we want
#        offset(k, n) = k * b_row_stride + n
#    So:
BK, BN = 3, 4
b_row_stride = 10            # pretend B has 10 columns
k_offsets = torch.arange(BK)
n_offsets = torch.arange(BN)

b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :]
print("b_offsets shape:", b_offsets.shape)   # [BK, BN]
print("b_offsets:\n", b_offsets)

# %%
# 4) "Loading" a 2-D block from a flat buffer — exactly what tl.load does.
#    Make a fake B (3 rows x 10 cols) and pull out a [BK x BN] block via flat indexing.
B = torch.arange(3 * 10).reshape(3, 10)
print("B:\n", B)

flat = B.flatten()
loaded = flat[b_offsets]                     # fancy indexing -> shape [BK, BN]
print("loaded block:\n", loaded)
print("matches B[:BK, :BN]?", torch.equal(loaded, B[:BK, :BN]))

# %%
# 5) Bounds masks. In Triton you can't have variable-length tensors, so you
#    over-fetch and zero out the OOB lanes via a mask. Same pattern in torch:
K_real = 2                                   # pretend the "real" K is only 2
k_mask = k_offsets < K_real                  # [BK]
print("k_mask:", k_mask)

# 2-D mask via broadcasting (this is what you'd pass to tl.load for a 2-D block)
n_mask = n_offsets < BN                      # [BN]  (trivially True here)
mask_2d = k_mask[:, None] & n_mask[None, :]  # [BK, BN]
print("mask_2d:\n", mask_2d)

masked_load = torch.where(mask_2d, loaded, torch.zeros_like(loaded))
print("masked_load:\n", masked_load)

# %%
# 6) The reduction shape question from your matmul.
#    a_vals: [BK]      (a row-slice of A)
#    b_vals: [BK, BN]  (a block of B)
#    want acc: [BN] s.t. acc[n] = sum_k a[k] * B[k, n]
a_vals = torch.tensor([1.0, 2.0, 3.0])                     # [BK=3]
b_vals = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)  # [BK=3, BN=4]

# Option A: broadcast then reduce
acc_A = (a_vals[:, None] * b_vals).sum(dim=0)              # [BN]
print("acc (broadcast+sum):", acc_A)

# Option B: a 1-D @ 2-D matmul (mirrors tl.dot)
acc_B = a_vals @ b_vals                                    # [BN]
print("acc (matmul):       ", acc_B)

# %%
# 7) Tiny real Triton kernel — useful for sanity-checking that your offset math
#    actually does what you think on the GPU. Uses tl.device_print to dump a
#    couple of values per program. Run this cell and watch stdout.
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def play_kernel(
    b_ptr,
    out_ptr,
    b_row_stride,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    k_offsets = tl.arange(0, BK)
    n_offsets = tl.arange(0, BN)

    # The exact pattern you need for the row-major matmul:
    b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :]   # [BK, BN]

    b_vals = tl.load(b_ptr + b_offsets)                                  # [BK, BN]

    # Fake "a row" so we can exercise the reduction shape too.
    a_vals = tl.arange(0, BK).to(tl.float32) + 1.0                       # [BK]
    acc = tl.sum(a_vals[:, None] * b_vals, axis=0)                       # [BN]

    tl.store(out_ptr + n_offsets, acc)

    # Peek at a few values (only program 0 prints, since we launch a 1-program grid).
    tl.device_print("b[0,0] = ", tl.load(b_ptr + 0))
    tl.device_print("acc[0] = ", acc[0])


BK, BN = 4, 8
B = torch.arange(BK * BN, dtype=torch.float32, device=DEVICE).reshape(BK, BN)
out = torch.empty(BN, device=DEVICE, dtype=torch.float32)

play_kernel[(1,)](B, out, B.stride(0), BK, BN)
torch.cuda.synchronize()

# Reference using pure torch:
a = torch.arange(BK, dtype=torch.float32, device=DEVICE) + 1.0
ref = a @ B
print("triton out:", out)
print("torch  ref:", ref)
print("match?    ", torch.allclose(out, ref))

# %%
