# %%
# pyright: reportUnreachable=false
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

@triton.jit
def preprocess_kernel(
    O_ptr, dO_ptr, #[N_queries, d], [N_queries, d]
    D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dOb, stride_dOq, stride_dOd,
    stride_db, stride_dq,
    N_QUERIES,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dOb,
        shape=(N_QUERIES, d),
        strides=(stride_dOq, stride_dOd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Oi = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    D = tl.sum(Oi * dOi, axis=-1)
    tl.store(D_block_ptr, D, boundary_check=(0,))

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, #[N_queries, d], [N_keys, d], [N_keys, d]
    dO_ptr, D_ptr, L_ptr, #[N_queries, d], [N_queries,] [N_queries,]
    dQ_ptr, # [N_queries, d]
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dOb, stride_dOq, stride_dOd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dQb, stride_dQq, stride_dQd,
    N_QUERIES, N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dOb,
        shape=(N_QUERIES, d),
        strides=(stride_dOq, stride_dOd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    # Triton miscompiles when the same loaded tile is used as both `X.T` and
    # `X` operands of tl.dot within the same loop iteration. We need K both
    # transposed (for Sij = Qi @ Kj.T) and non-transposed (for dQi += dSij @ Kj),
    # so we set up a separate transposed view (d, K_TILE_SIZE) and load both.
    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(d, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(d, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dQb,
        shape=(N_QUERIES, d),
        strides=(stride_dQq, stride_dQd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

    dQi = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)
        Kj_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero") # (d, K_TILE_SIZE)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)
        Sij = tl.dot(Qi, Kj_T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        # Mask padded key positions so they don't contribute to softmax.
        # Kj/Vj are zero-padded, so without this mask exp(0)=1 wousld leak weight.
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        if is_causal:
            # for causal attention, we need to mask the future keys
            # so, we only keep q >= k positions
            q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_offsets[:, None] >= k_offsets[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
            Sij = tl.where(mask, Sij, -float('inf'))
        Pij = tl.exp(Sij - Li[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dPij = tl.dot(dOi, Vj.T) # (Q_TILE_SIZE, K_TILE_SIZE)
        dSij = Pij * (dPij - Di[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        # dSij is fp32 (from exp); Kj is the input dtype. tl.dot needs both
        # to match, so cast dSij down to Kj's dtype before the matmul.
        dQi = dQi + tl.dot(dSij.to(Kj.dtype), Kj) * scale # (Q_TILE_SIZE, d)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        K_T_block_ptr = tl.advance(K_T_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))
    

@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, D_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dOb, stride_dOq, stride_dOd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dKb, stride_dKq, stride_dKd,
    stride_dVb, stride_dVq, stride_dVd,
    N_QUERIES, N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dOb,
        shape=(N_QUERIES, d),
        strides=(stride_dOq, stride_dOd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(d, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, key_tile_index * K_TILE_SIZE),
        block_shape=(d, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dKb,
        shape=(N_KEYS, d),
        strides=(stride_dKq, stride_dKd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dVb,
        shape=(N_KEYS, d),
        strides=(stride_dVq, stride_dVd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)
    Kj_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero") # (d, K_TILE_SIZE)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)

    k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    dKj = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dVj = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)

    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for i in range(Tq):
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

        Sij = tl.dot(Qi, Kj_T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        if is_causal:
            # for causal attention, we need to mask the future keys
            # so, we only keep q >= k positions
            q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_offsets[:, None] >= k_offsets[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
            Sij = tl.where(mask, Sij, -float('inf'))
        Pij = tl.exp(Sij - Li[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dVj = dVj + tl.dot(Pij.T, dOi) # (K_TILE_SIZE, d)
        dPij = tl.dot(dOi, Vj.T) # (Q_TILE_SIZE, K_TILE_SIZE)
        dSij = Pij * (dPij - Di[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dKj = dKj + tl.dot(tl.trans(dSij).to(Kj.dtype), Qi) * scale # (K_TILE_SIZE, d)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dKj.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))
    


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq, Bk = 16, 16
        B, N_queries, D = Q.shape
        N_keys = K.shape[1]
        scale = 1.0 / math.sqrt(D)
        Tq = triton.cdiv(N_queries, Bq)
        O = torch.empty((B, N_queries, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, N_queries), device=Q.device, dtype=torch.float32)
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
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Bq = Bq
        ctx.Bk = Bk
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        Bq, Bk = ctx.Bq, ctx.Bk

        B, N_queries, D = Q.shape
        N_keys = K.shape[1]
        Tq = triton.cdiv(N_queries, Bq)
        Tk = triton.cdiv(N_keys, Bk)

        # dO may arrive non-contiguous (e.g. from a slice upstream); the kernels
        # assume the usual (b, q, d) stride pattern, so make it contiguous.
        dO = dO.contiguous()

        # Precompute Di = rowsum(O * dO). Shared by the dQ and dKV kernels.
        Di = torch.empty(B, N_queries, device=Q.device, dtype=torch.float32)
        preprocess_kernel[(Tq, B)](
            O, dO, Di,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            Di.stride(0), Di.stride(1),
            N_queries, D, Bq,
        )

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        flash_bwd_dq_kernel[(Tq, B)](
            Q, K, V,
            dO, Di, L,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            Di.stride(0), Di.stride(1),
            L.stride(0), L.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_queries, N_keys, scale, D, Bq, Bk, is_causal,
        )

        flash_bwd_dkv_kernel[(Tk, B)](
            Q, K, V,
            dO, Di, L,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            Di.stride(0), Di.stride(1),
            L.stride(0), L.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_queries, N_keys, scale, D, Bq, Bk, is_causal,
        )

        # backward must return one gradient per forward input:
        #   forward(Q, K, V, is_causal) -> (dQ, dK, dV, None)
        # is_causal is a Python bool, no gradient.
        return dQ, dK, dV, None


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


def check_preprocess_correctness():
    """Verify preprocess_kernel computes D = rowsum(O * dO).

    Exercises non-multiple-of-tile N_queries to confirm boundary handling
    (zero-padded loads contribute 0 to the rowsum; the store skips OOB rows).
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    Bq = 16

    configs = [
        # (B, N_queries, d)
        (1, 16, 32),
        (2, 32, 64),
        (4, 64, 64),
        (8, 128, 128),
        (2, 48, 64),   # non-multiple of Bq stays aligned, sanity check
        (3, 17, 32),   # boundary: 17 not a multiple of Bq=16
        (2, 33, 64),   # boundary: 33 = 2*16 + 1
    ]

    atol, rtol = 1e-5, 1e-5
    all_ok = True
    for B, Nq, d in configs:
        O = torch.randn(B, Nq, d, device=device, dtype=dtype)
        dO = torch.randn(B, Nq, d, device=device, dtype=dtype)
        D = torch.empty(B, Nq, device=device, dtype=dtype)

        Tq = triton.cdiv(Nq, Bq)
        preprocess_kernel[(Tq, B)](
            O, dO, D,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            Nq,
            d, Bq,
        )

        D_ref = (O * dO).sum(dim=-1)

        max_abs = (D - D_ref).abs().max().item()
        mean_abs = (D - D_ref).abs().mean().item()
        ok = torch.allclose(D, D_ref, atol=atol, rtol=rtol)
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] preprocess B={B} Nq={Nq} d={d} | "
            f"max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e}"
        )

    print("\nPreprocess:", "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


def check_dq_correctness():
    """Verify flash_bwd_dq_kernel computes dQ correctly against autograd.

    Pipeline per config:
      1. Run flash_fwd_kernel -> O, L
      2. Run preprocess_kernel -> D = rowsum(O * dO)
      3. Run flash_bwd_dq_kernel -> dQ
      4. Compare to torch.autograd.grad(naive_attention(...), Q, dO)
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    Bq, Bk = 16, 16

    configs = [
        # (B, N_queries, N_keys, D)
        (1, 16, 16, 32),
        (2, 32, 32, 64),
        (4, 64, 64, 64),
        (8, 128, 128, 128),
        (2, 48, 80, 64),
        (3, 17, 33, 32),
    ]

    atol, rtol = 1e-2, 1e-2
    all_ok = True
    for B, Nq, Nk, D in configs:
        Q = torch.randn(B, Nq, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, Nk, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, Nk, D, device=device, dtype=dtype, requires_grad=True)
        dO = torch.randn(B, Nq, D, device=device, dtype=dtype)

        scale = 1.0 / math.sqrt(D)
        Tq = triton.cdiv(Nq, Bq)

        for is_causal in (False, True):
            # Reference dQ via autograd on the naive attention.
            O_ref = naive_attention(Q, K, V, is_causal=is_causal)
            (dQ_ref,) = torch.autograd.grad(O_ref, Q, dO, retain_graph=False)

            # 1. Forward pass to get O and L for use in backward.
            O = torch.empty(B, Nq, D, device=device, dtype=dtype)
            L = torch.empty(B, Nq, device=device, dtype=dtype)
            flash_fwd_kernel[(Tq, B)](
                Q.detach(), K.detach(), V.detach(),
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk, scale, D, Bq, Bk, is_causal,
            )

            # 2. Preprocess: Di = rowsum(O * dO).
            Di = torch.empty(B, Nq, device=device, dtype=dtype)
            preprocess_kernel[(Tq, B)](
                O, dO, Di,
                O.stride(0), O.stride(1), O.stride(2),
                dO.stride(0), dO.stride(1), dO.stride(2),
                Di.stride(0), Di.stride(1),
                Nq, D, Bq,
            )

            # 3. dQ backward kernel.
            dQ = torch.empty(B, Nq, D, device=device, dtype=dtype)
            flash_bwd_dq_kernel[(Tq, B)](
                Q.detach(), K.detach(), V.detach(),
                dO, Di, L,
                dQ,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                dO.stride(0), dO.stride(1), dO.stride(2),
                Di.stride(0), Di.stride(1),
                L.stride(0), L.stride(1),
                dQ.stride(0), dQ.stride(1), dQ.stride(2),
                Nq, Nk, scale, D, Bq, Bk, is_causal,
            )

            max_abs = (dQ - dQ_ref).abs().max().item()
            mean_abs = (dQ - dQ_ref).abs().mean().item()
            ok = torch.allclose(dQ, dQ_ref, atol=atol, rtol=rtol)
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] dQ causal={is_causal} B={B} Nq={Nq} Nk={Nk} D={D} | "
                f"max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e}"
            )

    print("\ndQ:", "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


def check_dkv_correctness():
    """Verify flash_bwd_dkv_kernel computes dK, dV against autograd.

    Pipeline per config:
      1. Run flash_fwd_kernel -> O, L
      2. Run preprocess_kernel -> Di = rowsum(O * dO)
      3. Run flash_bwd_dkv_kernel -> dK, dV
      4. Compare to torch.autograd.grad(naive_attention(...), (K, V), dO)
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    Bq, Bk = 16, 16

    configs = [
        # (B, N_queries, N_keys, D)
        (1, 16, 16, 32),
        (2, 32, 32, 64),
        (4, 64, 64, 64),
        (8, 128, 128, 128),
        (2, 48, 80, 64),
        (3, 17, 33, 32),
    ]

    atol, rtol = 1e-2, 1e-2
    all_ok = True
    for B, Nq, Nk, D in configs:
        Q = torch.randn(B, Nq, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, Nk, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, Nk, D, device=device, dtype=dtype, requires_grad=True)
        dO = torch.randn(B, Nq, D, device=device, dtype=dtype)

        scale = 1.0 / math.sqrt(D)
        Tq = triton.cdiv(Nq, Bq)
        Tk = triton.cdiv(Nk, Bk)

        for is_causal in (False, True):
            # Reference dK, dV via autograd on the naive attention.
            O_ref = naive_attention(Q, K, V, is_causal=is_causal)
            dK_ref, dV_ref = torch.autograd.grad(
                O_ref, (K, V), dO, retain_graph=False
            )

            # 1. Forward pass to get O and L.
            O = torch.empty(B, Nq, D, device=device, dtype=dtype)
            L = torch.empty(B, Nq, device=device, dtype=dtype)
            flash_fwd_kernel[(Tq, B)](
                Q.detach(), K.detach(), V.detach(),
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk, scale, D, Bq, Bk, is_causal,
            )

            # 2. Preprocess: Di = rowsum(O * dO).
            Di = torch.empty(B, Nq, device=device, dtype=dtype)
            preprocess_kernel[(Tq, B)](
                O, dO, Di,
                O.stride(0), O.stride(1), O.stride(2),
                dO.stride(0), dO.stride(1), dO.stride(2),
                Di.stride(0), Di.stride(1),
                Nq, D, Bq,
            )

            # 3. dK, dV backward kernel.
            dK = torch.empty(B, Nk, D, device=device, dtype=dtype)
            dV = torch.empty(B, Nk, D, device=device, dtype=dtype)
            flash_bwd_dkv_kernel[(Tk, B)](
                Q.detach(), K.detach(), V.detach(),
                dO, Di, L,
                dK, dV,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                dO.stride(0), dO.stride(1), dO.stride(2),
                Di.stride(0), Di.stride(1),
                L.stride(0), L.stride(1),
                dK.stride(0), dK.stride(1), dK.stride(2),
                dV.stride(0), dV.stride(1), dV.stride(2),
                Nq, Nk, scale, D, Bq, Bk, is_causal,
            )

            max_dK = (dK - dK_ref).abs().max().item()
            max_dV = (dV - dV_ref).abs().max().item()
            ok_K = torch.allclose(dK, dK_ref, atol=atol, rtol=rtol)
            ok_V = torch.allclose(dV, dV_ref, atol=atol, rtol=rtol)
            ok = ok_K and ok_V
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            tag_K = " " if ok_K else "!"
            tag_V = " " if ok_V else "!"
            print(
                f"[{status}] dKV causal={is_causal} B={B} Nq={Nq} Nk={Nk} D={D} | "
                f"dK{tag_K}err={max_dK:.3e}  dV{tag_V}err={max_dV:.3e}"
            )

    print("\ndKV:", "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


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


def check_end_to_end_backward():
    """Verify the full FlashAttentionFunc.backward against autograd.

    Runs forward + backward through the autograd.Function and compares
    dQ, dK, dV in one shot against torch.autograd.grad on naive_attention.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    configs = [
        # (B, N_queries, N_keys, D)
        (1, 16, 16, 32),
        (2, 32, 32, 64),
        (4, 64, 64, 64),
        (8, 128, 128, 128),
        (2, 48, 80, 64),
        (3, 17, 33, 32),
    ]

    atol, rtol = 1e-2, 1e-2
    all_ok = True
    for B, Nq, Nk, D in configs:
        for is_causal in (False, True):
            Q = torch.randn(B, Nq, D, device=device, dtype=dtype, requires_grad=True)
            K = torch.randn(B, Nk, D, device=device, dtype=dtype, requires_grad=True)
            V = torch.randn(B, Nk, D, device=device, dtype=dtype, requires_grad=True)
            dO = torch.randn(B, Nq, D, device=device, dtype=dtype)

            # Reference: pure-PyTorch attention + autograd.
            O_ref = naive_attention(Q, K, V, is_causal=is_causal)
            dQ_ref, dK_ref, dV_ref = torch.autograd.grad(O_ref, (Q, K, V), dO)

            # Triton: forward + backward via the autograd.Function.
            Q2 = Q.detach().clone().requires_grad_(True)
            K2 = K.detach().clone().requires_grad_(True)
            V2 = V.detach().clone().requires_grad_(True)
            O_triton = FlashAttentionFunc.apply(Q2, K2, V2, is_causal)
            O_triton.backward(dO)

            max_dQ = (Q2.grad - dQ_ref).abs().max().item()
            max_dK = (K2.grad - dK_ref).abs().max().item()
            max_dV = (V2.grad - dV_ref).abs().max().item()
            ok = (
                torch.allclose(Q2.grad, dQ_ref, atol=atol, rtol=rtol)
                and torch.allclose(K2.grad, dK_ref, atol=atol, rtol=rtol)
                and torch.allclose(V2.grad, dV_ref, atol=atol, rtol=rtol)
            )
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] e2e causal={is_causal} B={B} Nq={Nq} Nk={Nk} D={D} | "
                f"dQ={max_dQ:.3e} dK={max_dK:.3e} dV={max_dV:.3e}"
            )

    print("\nE2E backward:", "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


if __name__ == "__main__":
    check_preprocess_correctness()
    print()
    check_forward_correctness()
    print()
    check_dq_correctness()
    print()
    check_dkv_correctness()
    print()
    check_end_to_end_backward()
# %%