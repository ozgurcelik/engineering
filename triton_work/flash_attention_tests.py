# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
import triton.testing

from flash_attention import (
    flash_attention_forward,
    flash_attention_forward_gqa,
    flash_attention_forward_gqa_autotuned,
    flash_attention_forward_kernel_autotuned,
    flash_attention_forward_kernel_tk_trick_autotuned,
    flash_attention_forward_stages_autotuned,
    naive_attention,
    pytorch_attention,
)

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
            "head_dim": 128,
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["sequence_length"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=[
            "flash_autotuned",
            "flash_tk_autotuned",
            "flash_stages_autotuned",
            "pytorch",
        ],
        line_names=[
            "flash_attention_forward_kernel + autotuned",
            "flash_attention_forward_kernel_tk_trick + autotuned",
            "flash_attention_forward_stages + autotuned",
            "PyTorch Official",
        ],
        styles=[
            ("orange", "-"),
            ("purple", "-"),
            ("blue", "-"),
            ("green", "-"),
        ],
        xlabel="Sequence length",
        ylabel="TFLOPs/sec",
        y_log=True,
        plot_name="causal_attention_flash_autotuned_variants_vs_pytorch_fp16",
        args={
            "batch_size": 4,
            "num_heads": 8,
            "head_dim": 128,
            "dtype": torch.float16,
        },
    )
)
def benchmark_causal_attention(
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

    def flash_autotuned():
        return flash_attention_forward(Q, K, V, is_causal=True, TK_trick=False, autotune=True)

    def flash_tk_autotuned():
        return flash_attention_forward(Q, K, V, is_causal=True, TK_trick=True, autotune=True)

    def flash_stages_autotuned():
        return flash_attention_forward(Q, K, V, is_causal=True, stages=True, autotune=True)

    def pytorch():
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Check causal outputs and complete autotuning before timing.
    pytorch_output = pytorch()
    for implementation in (flash_autotuned, flash_tk_autotuned, flash_stages_autotuned):
        flash_output, _ = implementation()
        torch.testing.assert_close(flash_output, pytorch_output, rtol=2e-2, atol=2e-2)

    if provider == "flash_autotuned":
        print(f"N={sequence_length}: {flash_attention_forward_kernel_autotuned.best_config}")
        ms = triton.testing.do_bench(flash_autotuned)
    elif provider == "flash_tk_autotuned":
        print(f"N={sequence_length}: {flash_attention_forward_kernel_tk_trick_autotuned.best_config}")
        ms = triton.testing.do_bench(flash_tk_autotuned)
    elif provider == "flash_stages_autotuned":
        print(f"N={sequence_length}: {flash_attention_forward_stages_autotuned.best_config}")
        ms = triton.testing.do_bench(flash_stages_autotuned)
    elif provider == "pytorch":
        ms = triton.testing.do_bench(pytorch)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Report the same useful causal FLOPs for all providers, regardless of
    # whether they skip fully masked key tiles using the TK trick.
    causal_pairs = sequence_length * (sequence_length + 1) // 2
    flops = 4 * batch_size * num_heads * causal_pairs * head_dim
    return flops * 1e-12 / (ms * 1e-3)


def plot_attention_results(result, is_causal=False, x_name="sequence_length", xlabel="Sequence Length"):
    """Plot benchmark results as grouped bars with values above each bar."""
    provider_colors = {
        "Standard": "tab:blue",
        "Triton Flash": "tab:orange",
        "flash_attention_forward_kernel + autotuned": "tab:orange",
        "flash_attention_forward_kernel_tk_trick + autotuned": "tab:purple",
        "flash_attention_forward_stages + autotuned": "tab:cyan",
        "Triton Flash GQA": "tab:orange",
        "Triton Flash GQA + autotuned": "tab:purple",
        "PyTorch Official": "tab:green",
    }

    plt.close("all")
    x_vals = result[x_name].astype(int)
    providers = [column for column in result.columns if column != x_name]
    x_positions = np.arange(len(x_vals))
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
    attention_name = "Causal attention" if is_causal else "Attention"
    ax.set_title(f"{attention_name} TFLOPs/sec: {provider_names} (FP16)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("TFLOPs/sec")
    ax.set_xticks(x_positions, x_vals)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.margins(y=0.1)
    ax.legend()
    fig.tight_layout()

    plt.show()


# %%
# GQA correctness: native Triton GQA vs PyTorch SDPA with enable_gqa.
gqa_batch_size = 2
gqa_num_q_heads = 16
gqa_sequence_length = 64
gqa_head_dim = 32

for gqa_num_kv_heads in (1, 2, 4, 8, 16):
    Q_gqa = torch.randn(
        gqa_batch_size, gqa_num_q_heads, gqa_sequence_length, gqa_head_dim,
        device=DEVICE, dtype=torch.bfloat16,
    )
    K_gqa = torch.randn(
        gqa_batch_size, gqa_num_kv_heads, gqa_sequence_length, gqa_head_dim,
        device=DEVICE, dtype=torch.bfloat16,
    )
    V_gqa = torch.randn(
        gqa_batch_size, gqa_num_kv_heads, gqa_sequence_length, gqa_head_dim,
        device=DEVICE, dtype=torch.bfloat16,
    )

    for is_causal in (False, True):
        flash_gqa_output, _ = flash_attention_forward_gqa(
            Q_gqa, K_gqa, V_gqa, is_causal=is_causal,
        )
        pytorch_gqa_output = torch.nn.functional.scaled_dot_product_attention(
            Q_gqa, K_gqa, V_gqa, is_causal=is_causal, enable_gqa=True,
        )
        max_abs = (flash_gqa_output - pytorch_gqa_output).abs().max().item()
        print(
            f"GQA Hq={gqa_num_q_heads} Hk={gqa_num_kv_heads} "
            f"causal={is_causal}: max abs diff={max_abs:.3e}"
        )
        torch.testing.assert_close(
            flash_gqa_output, pytorch_gqa_output, rtol=2e-2, atol=2e-2,
        )

print("GQA correctness checks passed.")


# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_kv_heads"],
        x_vals=[1, 2, 4, 8, 16],
        line_arg="provider",
        line_vals=["flash_gqa", "flash_gqa_autotuned", "pytorch"],
        line_names=[
            "Triton Flash GQA",
            "Triton Flash GQA + autotuned",
            "PyTorch Official",
        ],
        styles=[("orange", "-"), ("purple", "-"), ("green", "-")],
        xlabel="KV heads",
        ylabel="TFLOPs/sec",
        y_log=False,
        plot_name="gqa_attention_flash_vs_pytorch_fp16_n8192",
        args={
            "batch_size": 4,
            "num_q_heads": 16,
            "sequence_length": 8192,
            "head_dim": 128,
            "dtype": torch.float16,
            "is_causal": True,
        },
    )
)
def benchmark_gqa_attention(
    num_kv_heads,
    provider,
    batch_size,
    num_q_heads,
    sequence_length,
    head_dim,
    dtype,
    is_causal,
):
    Q = torch.randn(
        batch_size, num_q_heads, sequence_length, head_dim,
        device=DEVICE, dtype=dtype,
    )
    K = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_dim,
        device=DEVICE, dtype=dtype,
    )
    V = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_dim,
        device=DEVICE, dtype=dtype,
    )

    def flash_gqa():
        return flash_attention_forward_gqa(Q, K, V, is_causal=is_causal)

    def flash_gqa_autotuned():
        return flash_attention_forward_gqa(Q, K, V, is_causal=is_causal, autotune=True)

    def pytorch():
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal, enable_gqa=True,
        )

    # Correctness check and autotune warm-up before timing.
    pytorch_output = pytorch()
    for implementation in (flash_gqa, flash_gqa_autotuned):
        flash_output, _ = implementation()
        torch.testing.assert_close(flash_output, pytorch_output, rtol=2e-2, atol=2e-2)

    if provider == "flash_gqa":
        ms = triton.testing.do_bench(flash_gqa)
    elif provider == "flash_gqa_autotuned":
        print(f"Hk={num_kv_heads}: {flash_attention_forward_gqa_autotuned.best_config}")
        ms = triton.testing.do_bench(flash_gqa_autotuned)
    elif provider == "pytorch":
        ms = triton.testing.do_bench(pytorch)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Useful causal FLOPs still scale with query heads, not KV heads.
    causal_pairs = sequence_length * (sequence_length + 1) // 2
    flops = 4 * batch_size * num_q_heads * causal_pairs * head_dim
    return flops * 1e-12 / (ms * 1e-3)


# %%

benchmark_results = benchmark_attention.run(print_data=True, return_df=True)
plot_attention_results(benchmark_results)
# %%

causal_benchmark_results = benchmark_causal_attention.run(print_data=True, return_df=True)
plot_attention_results(causal_benchmark_results, is_causal=True)

# %%

gqa_benchmark_results = benchmark_gqa_attention.run(print_data=True, return_df=True)
plot_attention_results(
    gqa_benchmark_results,
    is_causal=True,
    x_name="num_kv_heads",
    xlabel="KV Heads (Hq=16, N=8192)",
)

# %%
