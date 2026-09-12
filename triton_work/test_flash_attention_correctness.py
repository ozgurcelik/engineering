"""GPU correctness checks, independent of the plotting/benchmark notebook."""

import math

import pytest
import torch
from triton.runtime.errors import OutOfResources

import flash_attention as attention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("config", attention.flash_attention_forward_kernel_autotuned.configs)
def test_causal_config(config, monkeypatch):
    # Force each candidate: checking only the winner can miss an invalid tile.
    class FixedConfig:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return attention.flash_attention_forward_kernel[grid](
                    *args, **kwargs, **config.all_kwargs()
                )
            return launch

    monkeypatch.setattr(attention, "flash_attention_forward_kernel_autotuned", FixedConfig())
    try:
        check_attention(torch.float16, True, True, autotune=True)
    except OutOfResources as exc:
        # The autotuner also rejects candidates that exceed this GPU's limits.
        pytest.skip(str(exc))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal,tk", [(False, True), (True, False), (True, True)])
def test_fixed_attention(dtype, causal, tk):
    check_attention(dtype, causal, tk)


def check_attention(dtype, causal, tk, autotune=False):
    torch.manual_seed(42)
    # Unequal sequence lengths and partial tiles exercise both boundary masks.
    q = torch.randn((2, 2, 137, 96), device="cuda", dtype=dtype)
    k, v = [torch.randn((2, 2, 173, 96), device="cuda", dtype=dtype) for _ in range(2)]
    output, logsumexp = attention.flash_attention_forward(
        q, k, v, is_causal=causal, TK_trick=tk, autotune=autotune
    )
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.arange(q.shape[-2], device="cuda")[:, None] >= torch.arange(k.shape[-2], device="cuda")
        scores.masked_fill_(~mask, -float("inf"))
    expected = scores.softmax(-1) @ v.float()
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    # L must retain natural-log units after switching the internal exp to exp2.
    torch.testing.assert_close(logsumexp, scores.logsumexp(-1), atol=2e-4, rtol=2e-4)
