# %%
# pyright: reportUnreachable=false
import triton
import triton.language as tl
import torch
from torch import Tensor
import math
from einops import einsum
from jaxtyping import Bool, Float
from typing import Tuple
# %%
def naive_attention(Q: Float[Tensor, " ... L d_h"],
                    K: Float[Tensor, " ... L d_k"],
                    V: Float[Tensor, " ... L d_k"],
                    mask: Bool[Tensor, " ... L L"] | None = None) -> Float[Tensor, " ... L d_k"]:
    """
    Naive attention implementation.
    """
    d_k = K.shape[-1]
    scores = einsum(Q, K, "... query d, ... key d -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return einsum(weights, V, "... query key, ... key d -> ... query d")

def pytorch_attention(Q: Float[Tensor, " ... L d_h"],
                      K: Float[Tensor, " ... L d_k"],
                      V: Float[Tensor, " ... L d_k"],
                      mask: Bool[Tensor, " ... L L"] | None = None) -> Float[Tensor, " ... L d_k"]:
    """
    PyTorch attention implementation.
    """
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

# %%
@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, #[B, Hq, Lq, D]
    K_ptr, #[B, Hk, Lk, D]
    V_ptr, #[B, Hk, Lk, D]
    O_ptr, #[B, Hq, Lq, D]
    L_ptr, #[B, Hq, Lq]
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_lb, stride_lh, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    TK_trick: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    head_index = tl.program_id(2)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # we will loop over the entire key matrix
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.zeros((Q_TILE_SIZE, triton.next_power_of_2(D)), dtype=tl.float32) # (Q_TILE_SIZE, D)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (Q_TILE_SIZE,)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32) # (Q_TILE_SIZE,)
    
    # For causal attention, key tiles whose smallest key index is already past
    # the largest query index in this query tile would have Sij entirely masked
    # to -inf -- which is a no-op for the running (mi, li, Oi) state but still
    # costs two tl.loads, a tl.dot, and the mask/exp work. Tightening the loop
    # bound to skip those tiles cuts work roughly in half for causal and is
    # what makes causal attention actually faster than full attention.
    if is_causal and TK_trick:
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
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        # Scale by log2(e) so the softmax update can use exp2 directly.
        Sij = tl.dot(Qi, Kj.T) * (scale * 1.4426950408889634) # (Q_TILE_SIZE, K_TILE_SIZE)
        # Due to padding, some parts of the Sij matrix will be 0
        # But, this would mess up the softmax operation,
        # so we need to identify the padded elements and set the corresponding elements of Sij to -inf
        # now, the padding from the 0th dimension of Q is irrelevant since softmax is apllied along for each row separately, so any extra rows in S will be ignored down the line anyways
        # the padding along the 1st dimension of Q and K is also irrelevant since we will be multiplying all the elements along the D dimension of Q and K, so padded parts will be adding 0 to summation
        # But we need to take care of the padding along the 0th dimension of K, since padded parts there will be adding columns filled with 0s to the Sij matrix, which then messes up the denominator of softmax
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        if is_causal:
            if TK_trick:
                # Only tiles overlapping this query block need a causal mask.
                # Compare positions so rectangular query/key tiles work too.
                if (j + 1) * K_TILE_SIZE > query_tile_index * Q_TILE_SIZE:
                    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) # (Q_TILE_SIZE,)
                    mask = q_offsets[:, None] >= k_offsets[None, :] # (Q_TILE_SIZE, K_TILE_SIZE)
                    Sij = tl.where(mask, Sij, -float('inf'))
            else:
                q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) # (Q_TILE_SIZE,)
                mask = q_offsets[:, None] >= k_offsets[None, :] # (Q_TILE_SIZE, K_TILE_SIZE)
                Sij = tl.where(mask, Sij, -float('inf'))
        mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
        Pij = tl.exp2(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        alpha = tl.exp2(mi - mi_new)
        li = alpha * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
        Oi = tl.dot(Pij.to(Vj.dtype), Vj, Oi * alpha[:, None]) # (Q_TILE_SIZE, D)
        mi = mi_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    Oi = Oi * (1.0 / li[:, None])
    # Convert the base-2 softmax state back to natural-log logsumexp.
    Li = (mi + tl.log2(li)) * 0.6931471805599453
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

# Include the fixed 32x32 / 4-warp / 3-stage baseline and larger tiles
# to improve reuse of K/V across query rows.
flash_attention_forward_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"Q_TILE_SIZE": q_tile, "K_TILE_SIZE": k_tile},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for q_tile, k_tile in ((32, 32), (64, 32), (64, 64), (128, 32), (128, 64), (128, 128))
        for num_warps in (4, 8)
        for num_stages in (2, 3)
    ],
    key=["N_QUERIES", "N_KEYS", "D", "is_causal", "TK_trick"],
)(flash_attention_forward_kernel)


def flash_attention_forward(Q: Float[Tensor, " ... L d_h"],
                            K: Float[Tensor, " ... L d_k"],
                            V: Float[Tensor, " ... L d_k"],
                            is_causal: bool = False,
                            TK_trick: bool = True,
                            autotune: bool = False) -> Tuple[Float[Tensor, " ... L d_k"], Float[Tensor, " ... L"]]:
    """
    Flash attention forward pass.
    """
    B, Hq, Lq, D = Q.shape
    B, Hk, Lk, D = K.shape
    B, Hk, Lk, D = V.shape
    scale = 1.0 / math.sqrt(D)
    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32
    O = torch.empty((B, Hq, Lq, D), device=Q.device, dtype=Q.dtype)
    L = torch.empty((B, Hq, Lq), device=Q.device, dtype=torch.float32)
    kernel = flash_attention_forward_kernel_autotuned if autotune else flash_attention_forward_kernel
    grid = lambda meta: (triton.cdiv(Lq, meta["Q_TILE_SIZE"]), B, Hq)
    tile_args = {} if autotune else {
        "Q_TILE_SIZE": Q_TILE_SIZE,
        "K_TILE_SIZE": K_TILE_SIZE,
        "num_warps": 4,
        "num_stages": 3,
    }
    kernel[grid](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        Lq, Lk,
        scale,
        D,
        is_causal=is_causal, TK_trick=TK_trick, **tile_args,
    )
    return O, L

# %%
# TODO: Better grouping the programs
# TODO: Masking the Tk if we know it wont be used
