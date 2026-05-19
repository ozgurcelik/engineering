# %%
import torch
import math

# %%
Nq = 1024
Nk = 1024
d = 128
Bq = 16
Bk = 16
# %%
Q = torch.randn(Nq, d) # [Nq, d]
K = torch.randn(Nk, d) # [Nk, d]
V = torch.randn(Nk, d) # [Nk, d]
# %%
O = torch.zeros(Nq, d) # [Nq, d]
L = torch.zeros(Nq) # [Nq] logsumexp

for i in range(math.ceil(Nq / Bq)):
    Qi = Q[i * Bq:(i + 1) * Bq, :] # [Bq, d]
    Oi = torch.zeros(Bq, d) # [Bq, d]
    li = torch.zeros(Bq) # [Bq]
    mi = -torch.ones(Bq) * float('inf') # [Bq]
    for j in range(math.ceil(Nk / Bk)):
        Kj = K[j * Bk:(j + 1) * Bk, :] # [Bk, d]
        Vj = V[j * Bk:(j + 1) * Bk, :] # [Bk, d]
        Sij = Qi @ Kj.T / math.sqrt(d) # [Bq, Bk]
        mi_new = torch.max(mi, Sij.max(dim=1).values) # [Bq]
        Pij = torch.exp(Sij - mi_new[:, None]) # [Bq, Bk]
        alpha = torch.exp(mi - mi_new) # [Bq]
        li = alpha * li + Pij.sum(dim=1) # [Bq]
        Oi = alpha[:, None] * Oi + Pij @ Vj # [Bq, d]
        mi = mi_new
    Oi = Oi / li[:, None] # [Bq, d]
    Li = mi + torch.log(li) # [Bq]
    O[i * Bq:(i + 1) * Bq, :] = Oi
    L[i * Bq:(i + 1) * Bq] = Li

# %%
# Reference: standard attention for correctness check
S_ref = Q @ K.T / math.sqrt(d) # [Nq, Nk]
P_ref = torch.softmax(S_ref, dim=-1) # [Nq, Nk]
O_ref = P_ref @ V # [Nq, d]
L_ref = torch.logsumexp(S_ref, dim=-1) # [Nq]

print("max |O - O_ref|:", (O - O_ref).abs().max().item())
print("max |L - L_ref|:", (L - L_ref).abs().max().item())