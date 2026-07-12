# FSDP From Scratch

## Introduction

The purpose of this project is to learn how FSDP works and implement it from scratch.

## Why FSDP?

FSDP stands for Fully Sharded Data Parallel.
Data parallel part refers to the data parallelism in the algorithm.
So, in each rank, we process a different subset of the data, and in that sense, it is works like the data parallel.
But one problem with the data parallellism is that each rank holds a full copy of the model, does a full forward and backward pass, and then syncs the gradients before the optimizer step.
This means that if a GPU cannot fit the model in its memory, then the data parallel approach is not feasible.
FSDP solves this problem by sharding the model across the ranks so that no rank needs to hold a full copy of the model.
Since FSDP shards the model parameters, we also shard the gradients and optimizer states across the ranks.

## How does FSDP work?

As we run the FSDP, each GPU will hold a shard of the model parameters.
But to be able to do the forward and backward pass through a layer in a GPU, we will need all the parameters for a layer materialized in the GPU.
This requires us to do an all-gather operation before the forward pass.
Once the forward pass is done, we can free the memory of the parameters.
This means, during the forward pass, we can get away with fully holding the parameters of only one layer at a time.
During the backward pass, we will need all the parameters for the layer that we are currently processing once again.
Why is that?
Imagine we have a linear layer with parameters `W` and input `x` generating output `y`.
The forward pass is given by `y = Wx`.
And we have the loss function `L` with `dL/dy` available to us.
Then, the gradients for this layer is given by `dL/dW = dL/dy * x^T` and this does not need us to have the full `W` in memory.
But, the input gradients `dL/dx = W^T * dL/dy` does need us to have the full `W` in memory.