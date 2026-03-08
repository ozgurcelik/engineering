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

    # All ranks have the same full data (unlike data parallelism, where data is sharded).
    # In tensor parallelism, we shard the *model weights* instead.
    data = generate_sample_data()
    batch_size = data.shape[0]
    num_dim = data.shape[1]

    # Column-wise sharding: the full weight matrix for each layer is [num_dim, num_dim].
    # We split it by columns across ranks, so each rank holds [num_dim, local_num_dim].
    #
    #   Full A [num_dim, num_dim] = [ A_rank0 | A_rank1 | A_rank2 | A_rank3 ]
    #                                [num_dim, local] each
    #
    # Each rank stores a DIFFERENT column shard, so params will diverge across ranks.
    # (Compare to data parallelism, where every rank has the same full weight matrix.)
    local_num_dim = num_dim // world_size
    params = [get_init_params(num_dim, local_num_dim, rank) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=0.001)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        # We store layer_inputs[i] = the input to layer i (shape [batch, num_dim]).
        # These are needed during the backward pass to reconstruct the computation graph.
        # This is the same memory cost as standard autograd (which also stores activations
        # internally). It could be reduced via activation checkpointing (recompute instead
        # of store), but we skip that optimization here for clarity.
        layer_inputs = []

        x = data
        for layer in range(num_layers):
            # Save the input to this layer for the backward pass.
            layer_inputs.append(x.detach())

            # Each rank computes its local output: x @ A_rank
            #   x:            [batch, num_dim]      (same on all ranks)
            #   params[layer]: [num_dim, local_num_dim]  (different on each rank)
            #   result:        [batch, local_num_dim]    (different on each rank)
            x = x @ params[layer]
            x = F.relu(x)

            # All-gather: collect local outputs from every rank and concatenate them
            # so that each rank has the full [batch, num_dim] tensor for the next layer.
            #
            # Forward communication pattern:
            #   rank 0: [batch, local_dim] ──┐
            #   rank 1: [batch, local_dim] ──┼──→ all_gather ──→ [batch, num_dim] (on every rank)
            #   rank 2: [batch, local_dim] ──┤
            #   rank 3: [batch, local_dim] ──┘
            activations = [torch.empty(batch_size, local_num_dim) for _ in range(world_size)]
            dist.all_gather(tensor_list=activations, tensor=x.detach(), async_op=False)
            x = torch.cat(activations, dim=1)  # [batch, num_dim]

        loss = x.square().sum()

        # Backward pass
        #
        # We iterate from the last layer to the first. At each step we hold `grad`:
        # the gradient of the loss w.r.t. the gathered output of this layer [batch, num_dim].
        #
        # For each layer, we:
        #   1. Reconstruct the local forward (inp -> matmul -> relu -> x)
        #   2. Build a scalar `subloss` whose .backward() produces the correct gradients
        #   3. Call subloss.backward() to get params[layer].grad and inp.grad via autograd
        #   4. All-reduce inp.grad (partial -> full input gradient)
        grad = None

        for layer in range(num_layers - 1, -1, -1):

            # Create an isolated computation graph for just this layer.
            # detach() severs connection to any previous graph.
            # requires_grad_(True) tells autograd we want dL/d(inp).
            inp = layer_inputs[layer].detach().requires_grad_(True)
            inp.retain_grad()

            # Reconstruct the local forward pass for this layer.
            # This builds a small autograd graph: inp -> matmul -> relu -> x
            x = inp @ params[layer]  # [batch, num_dim] @ [num_dim, local_dim] -> [batch, local_dim]
            x = F.relu(x)

            if layer == num_layers - 1:
                # Last layer: use the actual loss function.
                # This works because loss = x_gathered.square().sum() decomposes over ranks:
                #   loss = rank0_output.square().sum() + rank1_output.square().sum() + ...
                # Each rank's output occupies a non-overlapping slice of the gathered tensor,
                # so x.square().sum() on this rank gives the correct local gradients.
                subloss = x.square().sum()
            else:
                # Intermediate layers: use the linear approximation trick.
                #
                # `grad` is dL/d(gathered output of this layer), shape [batch, num_dim].
                # We slice it to get the gradient for this rank's local output (reverse of all_gather):
                #   grad = [ grad_rank0 | grad_rank1 | grad_rank2 | grad_rank3 ]
                #   grad_local = grad[:, rank*local_dim:(rank+1)*local_dim]
                #
                # Then subloss = (x * grad_local).sum() is a scalar whose gradient w.r.t.
                # any variable v is: d/dv[sum(x * g)] = sum(g * dx/dv), which is exactly
                # the chain rule: upstream gradient g dotted with the local Jacobian dx/dv.
                # So subloss.backward() produces the same gradients as x.backward(gradient=grad_local).
                grad_local = grad[:, rank * local_num_dim:(rank + 1) * local_num_dim]
                subloss = (x * grad_local).sum()

            # Autograd computes gradients through the local graph (relu and matmul).
            # After this call:
            #   - params[layer].grad is set: dL/d(A_rank) = inp^T @ grad_through_relu
            #     This is the COMPLETE gradient for this rank's weight shard. No communication
            #     needed because inp is the same on all ranks and grad_through_relu is local.
            #     (A_rank only affects this rank's local output, so dL/dA_rank has one term.)
            #
            #   - inp.grad is set: dL/d(inp) = grad_through_relu @ A_rank^T
            #     This is a PARTIAL gradient. The full gradient is a sum over all ranks:
            #       dL/dX = grad_0 @ A_0^T + grad_1 @ A_1^T + ... + grad_3 @ A_3^T
            #     Each rank computed one dense [batch, num_dim] term. We need all-reduce(SUM)
            #     to get the total. (inp feeds into all ranks' matmuls, so dL/d(inp) has
            #     one term per rank.)
            subloss.backward()

            # All-reduce: sum partial input gradients across ranks.
            # After this, `grad` holds the full dL/d(layer input), which is also
            # dL/d(previous layer's gathered output) -- used in the next iteration.
            grad = inp.grad.clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=False)

        optimizer.step()

        dist.barrier()
        print(f"[Step {step}] Rank {rank} has params {params[0][0][:3]}")
        dist.barrier()
        if rank == 0:
            print("--------------------------------")
        dist.barrier()

    cleanup()


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int, num_micro_batches: int):

    setup(rank, world_size)

    # get the data
    data = generate_sample_data()
    batch_size = data.shape[0]
    num_dim = data.shape[1]
    
    # all the ranks use all the data, so no need to shard the data
    # params are sharded across ranks by layer

    local_num_layers = num_layers // world_size
    params = [get_init_params(num_dim, num_dim, rank) for _ in range(local_num_layers)]
    optimizer = torch.optim.AdamW(params, lr=0.001)
    
    # data is actually split to micro-batches
    micro_batch_size = batch_size // num_micro_batches
    
    # for the first stage, the data is the original data
    # for the rest of the stages, the data is the output of the previous stage
    if rank == 0:
        micro_batches = [data[i * micro_batch_size:(i + 1) * micro_batch_size] for i in range(num_micro_batches)]
    else:
        micro_batches = [torch.empty(micro_batch_size, num_dim) for _ in range(num_micro_batches)]

    for step in range(num_steps):
        optimizer.zero_grad()

        # store the inputs for each stage for the backward pass
        stage_inputs = [torch.empty(micro_batch_size, num_dim) for _ in range(num_micro_batches)]

        # forward pass
        for i in range(num_micro_batches):
            x = micro_batches[i]
            # if we are not in the first stage, get the output of the previous stage
            if rank != 0:
                dist.recv(tensor=x, src=rank - 1)
            stage_inputs[i] = x

            for layer in range(local_num_layers):
                x = x @ params[layer]
                x = F.relu(x)

            # if we are not in the last stage, send the output to the next stage
            if rank != world_size - 1:
                dist.send(tensor=x, dst=rank + 1)

        # backward pass
        if rank == world_size - 1:
            loss = [stage_inputs[i].square().sum() for i in range(num_micro_batches)].sum()
            loss.backward()


            

    cleanup()

def main():
    world_size = 2
    #mp.spawn(collective_operations, args=(world_size,), nprocs=world_size, join=True)
    data = generate_sample_data()
    num_layers = 4
    num_steps = 2
    num_micro_batches = 4
    mp.spawn(pipeline_parallelism_main, args=(world_size, data, num_layers, num_steps, num_micro_batches), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()