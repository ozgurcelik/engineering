# %%
import triton
import triton.language as tl
import torch
import math

"""
Goal: fuse the three steps of attention
    S = Q @ K^T
    P = softmax(S)
    O = P @ V
into a single kernel. The naive version is three kernel launches and
the intermediates S and P are O(N^2), so they get fully materialized
in HBM and reread by the next launch. Fusing means Sij and Pij stay
in registers / SRAM and never touch HBM.

Why the inner loop over Tk:
Doing it "in one go" would require the full K of shape (N_keys, D)
and the full Sij of shape (Q_TILE_SIZE, N_keys) to live in registers/SRAM
simultaneously, which blows the on-chip budget for any realistic seq len.
So we can't run offline softmax (which needs all of Sij to take max ->
exp -> normalize). Instead we use *online* softmax: process one key
tile at a time and fold each chunk into a running (mi, li, Oi) state.
The Tk loop is the form that online recurrence takes.
"""

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # For causal attention, key tiles whose smallest key index is already past
    # the largest query index in this query tile would have Sij entirely masked
    # to -inf -- which is a no-op for the running (mi, li, Oi) state but still
    # costs two tl.loads, a tl.dot, and the mask/exp work. Tightening the loop
    # bound to skip those tiles cuts work roughly in half for causal and is
    # what makes causal attention actually faster than full attention.
    if is_causal:
        # Last reachable key index for this query tile is
        #   (query_tile_index + 1) * Q_TILE_SIZE - 1,
        # so the number of key tiles we need to visit is
        #   ceil(((query_tile_index + 1) * Q_TILE_SIZE) / K_TILE_SIZE).
        # Also clamp to the actual number of key tiles so we don't run past N_KEYS.
        Tk = tl.minimum(
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE),
            tl.cdiv(N_KEYS, K_TILE_SIZE),
        )
    else:
        Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Sij = tl.dot(Qi, Kj.T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        # Mask padded key positions so they don't contribute to softmax.
        # Kj/Vj are zero-padded, so without this mask exp(0)=1 would leak weight.
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        if is_causal:
            # for causal attention, we need to mask the future keys
            # so, we only keep q >= k positions
            q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_offsets[:, None] >= k_offsets[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
            Sij = tl.where(mask, Sij, -float('inf'))
        mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
        Pij = tl.exp(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        alpha = tl.exp(mi - mi_new) # (Q_TILE_SIZE,)
        li = alpha * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
        Oi = alpha[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj) # (Q_TILE_SIZE, D)
        mi = mi_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi * (1.0 / li[:, None])
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        ctx.save_for_backward(Q, K, V)
        ctx.is_causal = is_causal
        Bq, Bk = 16, 16
        B, N_queries, D = Q.shape
        N_keys = K.shape[1]
        scale = 1.0 / math.sqrt(D)
        Tq = triton.cdiv(N_queries, Bq)
        O = torch.empty((B, N_queries, D), device=Q.device)
        L = torch.empty((B, N_queries), device=Q.device)
        flash_fwd_kernel[(Tq, B)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_queries, N_keys,
            scale,
            D,
            Bq, Bk,
            is_causal,
        )
        ctx.save_for_backward(O, L)
        return O
    
    @staticmethod
    def backward(ctx):
        raise NotImplementedError("Backward pass is not implemented")


# %%
def naive_attention(Q, K, V, is_causal=False):
    """Reference scaled dot-product attention.

    When `is_causal` is True, position `k` is masked out for query `q` whenever
    `k > q` (matches the kernel's absolute-position convention).
    """
    d = Q.shape[-1]
    S = torch.einsum("bqd,bkd->bqk", Q, K) / math.sqrt(d)
    if is_causal:
        Nq, Nk = Q.shape[1], K.shape[1]
        q_idx = torch.arange(Nq, device=Q.device)
        k_idx = torch.arange(Nk, device=Q.device)
        causal_mask = q_idx[:, None] >= k_idx[None, :]  # (Nq, Nk)
        S = S.masked_fill(~causal_mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bqk,bkd->bqd", P, V)


def check_forward_correctness():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    configs = [
        # (B, N_queries, N_keys, D)
        (1, 16, 16, 32),
        (2, 32, 32, 64),
        (4, 64, 64, 64),
        (8, 128, 128, 128),
        (2, 48, 80, 64),   # non-square, non-multiple-of-tile shapes
        (3, 17, 33, 32),   # odd sizes to exercise boundary handling
    ]

    atol, rtol = 1e-2, 1e-2
    all_ok = True
    for B, Nq, Nk, D in configs:
        Q = torch.randn(B, Nq, D, device=device, dtype=dtype)
        K = torch.randn(B, Nk, D, device=device, dtype=dtype)
        V = torch.randn(B, Nk, D, device=device, dtype=dtype)

        for is_causal in (False, True):
            O_ref = naive_attention(Q, K, V, is_causal=is_causal)
            O_triton = FlashAttentionFunc.apply(Q, K, V, is_causal)

            max_abs = (O_ref - O_triton).abs().max().item()
            mean_abs = (O_ref - O_triton).abs().mean().item()
            ok = torch.allclose(O_ref, O_triton, atol=atol, rtol=rtol)
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] causal={is_causal} B={B} Nq={Nq} Nk={Nk} D={D} | "
                f"max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e}"
            )

    print("\nOverall:", "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


if __name__ == "__main__":
    check_forward_correctness()
# %%