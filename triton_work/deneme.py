# %%
for i in range(10):
    print(i)
# %%
for i in range(0, 10, 2):
    print(i)
# %%
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# %%
k_offsets = [0, 1]
matrix[k_offsets]
# %%
import numpy as np
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
k_offsets = [0, 1]
matrix[k_offsets]
# %%
import torch
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
k_offsets = torch.tensor([0, 1])
matrix[k_offsets]
# %%
