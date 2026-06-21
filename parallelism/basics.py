import torch.distributed as dist
import sys
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def collective_operations(rank, world_size):
    setup(rank, world_size)
    
    # all_reduce
    tensor = torch.tensor([0, 1, 2, 3]) + rank
    print(f"[Before all_reduce] Rank {rank} has tensor {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"[After all_reduce] Rank {rank} has tensor {tensor}")

    dist.barrier()
    if rank == 0:
        print("--------------------------------")
    dist.barrier()

    # reduce_scatter
    input = torch.tensor([0, 1, 2, 3], dtype=torch.float32) + rank
    output = torch.empty(1)
    print(f"[Before reduce_scatter] Rank {rank} has input {input}, output {output}")

    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"[After reduce_scatter] Rank {rank} has output {output}")

    dist.barrier()
    if rank == 0:
        print("--------------------------------")
    dist.barrier()

    # all_gather
    input = output
    output = torch.empty(world_size)

    print(f"[Before all_gather] Rank {rank} has input {input}, output {output}")
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"[After all_gather] Rank {rank} has output {output}")

    dist.barrier()
    if rank == 0:
        print("--------------------------------")
    dist.barrier()

    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(collective_operations, args=(world_size,), nprocs=world_size, join=True)