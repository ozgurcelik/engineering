# DDP Implementations

In `ddp.py`, we have four implementations of DDP:

1. Naive DDP
2. Naive DDP with flattened gradients
3. DDP with hooks
4. DDP with buckets

## Naive DDP

In naive DDP, we perform the all-reduce operation after the backward pass is completed. But this means we cycle between the backward pass and the all-reduce operation, which is not efficient.

Additionally, we perform one all-reduce operation for each tensor, which introduces:

- **Latency** — setting up, waiting for handshake and confirmation for each all-reduce operation.
- **Kernel launch overhead** — each all-reduce operation is a kernel launch.
- **Bandwidth underutilization** — tiny messages can't saturate the network.

## Naive DDP with Flattened Gradients

In naive DDP with flattened gradients, we flatten the gradients of all the parameters and perform one all-reduce operation on the flattened gradients.

This helps with the second issue, but still requires us to wait for the backward pass to complete before we can start the all-reduce operation.

## DDP with Hooks

In DDP with hooks, we register post-accumulate hooks for each parameter, and when the backward pass is completed, we call the `finish_gradient_synchronization` method to perform the all-reduce operation.

This helps with the first issue, but now we are back to performing one all-reduce operation for each tensor.

## DDP with Buckets

In DDP with buckets, we group the parameters into buckets (parameters that are close to each other in the LLM architecture), and perform one all-reduce operation for each bucket.

This helps with both the first and second issues.
