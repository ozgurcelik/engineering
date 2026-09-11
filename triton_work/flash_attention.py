# %%
# pyright: reportUnreachable=false
import triton
import triton.language as tl
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
import math
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
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
torch.manual_seed(0)

batch_size = 2
num_heads = 4
sequence_length = 32
head_dim = 16

Q = torch.randn(batch_size, num_heads, sequence_length, head_dim)
K = torch.randn(batch_size, num_heads, sequence_length, head_dim)
V = torch.randn(batch_size, num_heads, sequence_length, head_dim)

naive_output = naive_attention(Q, K, V)
pytorch_output = pytorch_attention(Q, K, V)

max_absolute_difference = (naive_output - pytorch_output).abs().max().item()
print(f"Maximum absolute difference: {max_absolute_difference:.3e}")
torch.testing.assert_close(naive_output, pytorch_output, rtol=1e-5, atol=1e-6)

# %%
DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
    
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Sij = tl.dot(Qi, Kj.T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        # Due to padding, some parts of the Sij matrix will be 0
        # But, this would mess up the softmax operation,
        # so we need to identify the padded elements and set the corresponding elements of Sij to -inf
        # now, the padding from the 0th dimension of Q is irrelevant since softmax is apllied along for each row separately, so any extra rows in S will be ignored down the line anyways
        # the padding along the 1st dimension of Q and K is also irrelevant since we will be multiplying all the elements along the D dimension of Q and K, so padded parts will be adding 0 to summation
        # But we need to take care of the padding along the 0th dimension of K, since padded parts there will be adding columns filled with 0s to the Sij matrix, which then messes up the denominator of softmax
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
        Pij = tl.exp(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        li = tl.exp(mi - mi_new) * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
        Oi = tl.exp(mi - mi_new)[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj) # (Q_TILE_SIZE, D)
        mi = mi_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    Oi = Oi * (1.0 / li[:, None])
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

def flash_attention_forward(Q: Float[Tensor, " ... L d_h"],
                            K: Float[Tensor, " ... L d_k"],
                            V: Float[Tensor, " ... L d_k"],
                            is_causal: bool = False) -> Tuple[Float[Tensor, " ... L d_k"], Float[Tensor, " ... L"]]:    
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
    Tq = triton.cdiv(Lq, Q_TILE_SIZE)
    grid_size = (Tq, B, Hq)
    flash_attention_forward_kernel[(Tq, B, Hq)](
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
        Q_TILE_SIZE, K_TILE_SIZE, is_causal,
    )
    return O, L

# %%
# Reuse the inputs from the naive-vs-PyTorch comparison as BF16 on the GPU.
Q_gpu, K_gpu, V_gpu = (tensor.to(device=DEVICE, dtype=torch.bfloat16) for tensor in (Q, K, V))
flash_output, _ = flash_attention_forward(Q_gpu, K_gpu, V_gpu, is_causal=False)
pytorch_gpu_output = pytorch_attention(Q_gpu, K_gpu, V_gpu)

max_absolute_difference = (flash_output - pytorch_gpu_output).abs().max().item()
print(f"Flash vs PyTorch maximum absolute difference: {max_absolute_difference:.3e}")
# Allow for BF16 rounding differences between the two implementations.
torch.testing.assert_close(flash_output, pytorch_gpu_output, rtol=2e-2, atol=2e-2)
# %%

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["sequence_length"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["naive", "flash", "pytorch"],
        line_names=["Standard", "Triton Flash", "PyTorch Official"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        xlabel="Sequence length",
        ylabel="TFLOPs/sec",
        y_log=True,
        plot_name="attention_naive_vs_flash_vs_pytorch_fp16",
        args={
            "batch_size": 4,
            "num_heads": 8,
            "head_dim": 96,
            "dtype": torch.float16,
        },
    )
)
def benchmark_attention(
    sequence_length,
    provider,
    batch_size,
    num_heads,
    head_dim,
    dtype,
):
    shape = (batch_size, num_heads, sequence_length, head_dim)
    Q = torch.randn(shape, device=DEVICE, dtype=dtype)
    K = torch.randn(shape, device=DEVICE, dtype=dtype)
    V = torch.randn(shape, device=DEVICE, dtype=dtype)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == "naive":
        ms = triton.testing.do_bench(lambda: naive_attention(Q, K, V))
    elif provider == "flash":
        ms = triton.testing.do_bench(lambda: flash_attention_forward(Q, K, V, is_causal=False))
    elif provider == "pytorch":
        ms = triton.testing.do_bench(lambda: pytorch_attention(Q, K, V))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Count the multiply and add in each of QK^T and softmax(QK^T)V.
    flops = 4 * batch_size * num_heads * sequence_length**2 * head_dim
    return flops * 1e-12 / (ms * 1e-3)


def plot_attention_results(result):
    """Plot benchmark results as grouped bars with values above each bar."""
    provider_colors = {
        "Standard": "tab:blue",
        "Triton Flash": "tab:orange",
        "PyTorch Official": "tab:green",
    }

    plt.close("all")
    sequence_lengths = result["sequence_length"].astype(int)
    providers = [column for column in result.columns if column != "sequence_length"]
    x_positions = np.arange(len(sequence_lengths))
    bar_width = 0.8 / len(providers)

    fig, ax = plt.subplots(figsize=(14, 7))
    for provider_index, provider in enumerate(providers):
        offset = (provider_index - (len(providers) - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset,
            result[provider],
            width=bar_width,
            label=provider,
            color=provider_colors.get(provider),
        )
        ax.bar_label(bars, fmt="%.2f", padding=3)

    provider_names = " vs ".join(providers)
    ax.set_title(f"Attention TFLOPs/sec: {provider_names} (FP16)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("TFLOPs/sec")
    ax.set_xticks(x_positions, sequence_lengths)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.margins(y=0.1)
    ax.legend()
    fig.tight_layout()

    plt.show()


benchmark_results = benchmark_attention.run(print_data=True, return_df=True)
plot_attention_results(benchmark_results)


# %%
# TODO: Better grouping the programs
# TODO: Masking the Tk if we know it wont be used