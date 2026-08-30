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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["sequence_length"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["naive", "pytorch"],
        line_names=["Standard", "PyTorch Official"],
        styles=[("blue", "-"), ("green", "-")],
        xlabel="Sequence length",
        ylabel="TFLOPs/sec",
        y_log=True,
        plot_name="attention_naive_vs_pytorch_fp16",
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
