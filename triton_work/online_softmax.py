# %%
import torch

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Naive softmax implementation.
    """
    z_max = x.max(dim=1, keepdim=True).values
    z = x - z_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator

def online_softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Online softmax implementation.
    """
    M, N = x.shape
    y = torch.zeros_like(x)
    # Keep column dim so broadcasting against (M, 1) stays (M, 1) instead of
    # collapsing/expanding to (M, M).
    z_max = torch.full((M, 1), float('-inf'), dtype=x.dtype)
    for i in range(N):
        z_max = torch.max(z_max, x[:, i:i+1])

    exp_sum = torch.zeros((M, 1), dtype=x.dtype)
    for i in range(N):
        exp_sum += torch.exp(x[:, i:i+1] - z_max)

    for i in range(N):
        y[:, i:i+1] = torch.exp(x[:, i:i+1] - z_max) / exp_sum

    return y

# %%
if __name__ == "__main__":
    torch.manual_seed(0)

    shapes = [(1, 16), (4, 128), (32, 1024), (128, 4096)]
    dtypes = [torch.float32, torch.float16]
    impls = [naive_softmax, online_softmax_torch]

    for dtype in dtypes:
        # Looser tolerances for fp16 due to reduced precision.
        atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
        for shape in shapes:
            x = torch.randn(shape, dtype=dtype)
            ref = torch.softmax(x, dim=1)

            for impl in impls:
                mine = impl(x)
                torch.testing.assert_close(mine, ref, atol=atol, rtol=rtol)

                max_abs_err = (mine - ref).abs().max().item()
                row_sums = mine.sum(dim=1)
                print(
                    f"[OK] impl={impl.__name__} dtype={dtype} shape={shape} "
                    f"max_abs_err={max_abs_err:.2e} "
                    f"row_sum_min={row_sums.min().item():.6f} "
                    f"row_sum_max={row_sums.max().item():.6f}"
                )

    print("All correctness checks passed.")

# %%
