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

def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data

def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs) / math.sqrt(num_outputs))

def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # get the data
    data = generate_sample_data()
    # get a slice of the data
    batch_size = data.shape[0]
    num_dim = data.shape[1]
    local_batch_size = batch_size // world_size
    local_data = data[rank * local_batch_size:(rank + 1) * local_batch_size]

    # get the params
    params = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=0.001)

    for step in range(num_steps):
        # forward pass
        x = local_data
        for layer in range(num_layers):
            x = x @ params[layer]
            x = F.relu(x)

        loss = x.square().sum()

        # backward pass
        loss.backward()

        # all-reduce all the gradients
        for param in params:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # update the params
        optimizer.step()

        dist.barrier()
        print(f"[Step {step}] Rank {rank} has params {params[0][0][:3]}")
        dist.barrier()
        if rank == 0:
            print("--------------------------------")
        dist.barrier()

def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # all the ranks have the same data
    data = generate_sample_data()
    batch_size = data.shape[0]
    num_dim = data.shape[1]
    
    # lets do column-wise sharding
    local_num_dim = num_dim // world_size
    params = [get_init_params(local_num_dim, local_num_dim, rank) for _ in range(num_layers)]
    
    # lets only do one step for now
    x = data

    # forward pass
    for layer in range(num_layers):
        x = x @ params[layer]
        x = F.relu(x)

        # lets gather all the activations from all the ranks
        activations = [torch.empty(batch_size, local_num_dim) for _ in range(world_size)]

        # all-gather the activations
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # concatenate the activations
        x = torch.cat(activations, dim=1)

    loss = x.square().sum()

    # backward pass
    

def main():
    world_size = 4
    #mp.spawn(collective_operations, args=(world_size,), nprocs=world_size, join=True)
    data = generate_sample_data()
    num_layers = 2
    num_steps = 10
    mp.spawn(data_parallelism_main, args=(world_size, data, num_layers, num_steps), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()