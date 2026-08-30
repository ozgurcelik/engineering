# %%
# pyright: reportUnreachable=false
import torch
import math
# %%
def naive_attention(Q, K, V, is_causal=False):
    """Reference scaled dot-product attention."""