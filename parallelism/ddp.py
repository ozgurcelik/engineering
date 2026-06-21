import torch.distributed as dist
import torch

class DDP(torch.nn.Module):
    """
    Naïve distributed data parallel container.

    At construction time, the parameters of rank 0 are broadcast to all
    other ranks so that every replica starts from identical weights. After
    each backward pass, ``finish_gradient_synchronization`` all-reduces the
    gradient of every parameter tensor (averaged across ranks), so that all
    replicas apply the same optimizer update and stay in sync.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # Make every replica start from the same parameters by broadcasting
        # rank 0's weights (and buffers) to all other ranks.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """All-reduce (average) each parameter gradient across all ranks.

        Call this after ``loss.backward()`` and before ``optimizer.step()``.
        """
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.div_(self.world_size)

    def finish_gradient_synchronization_flat(self):
        """All-reduce (average) each parameter gradient across all ranks.

        Call this after ``loss.backward()`` and before ``optimizer.step()``.
        """
        param_grads = [p for p in self.module.parameters() if p.grad is not None]
        grads = [p.grad for p in param_grads]
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
        flat_grads.div_(self.world_size)
        synced_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for i, param in enumerate(param_grads):
            param.grad.copy_(synced_grads[i])


class DDP_Hook(torch.nn.Module):
    """
    When does hook run? 
    During loss.backward(). 
    PyTorch walks the backward graph, and the instant each param's .grad finishes accumulating, it fires that param's post-accumulate hook → which launches the async all-reduce. 
    So the all-reduces get sprayed out throughout backward, overlapping with the remaining gradient computation. 
    That's the whole win.

    State of self.handles when backward() returns? 
    By the time backward() returns, every param's grad has been computed, so every hook has fired — self.handles holds one handle per param (fully populated). 
    And exactly as you said: some of those all-reduces have already completed, some are still in flight on the background stream. 
    That's precisely why finish_gradient_synchronization must .wait() on each one before averaging.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # Make every replica start from the same parameters by broadcasting
        # rank 0's weights (and buffers) to all other ranks.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        self.handles = []

        self._register_hooks()

    def _register_hooks(self):
        def _hook(param: torch.nn.Parameter):
            if self.world_size == 1:
                return
            if param.grad is None:
                return
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
            
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """All-reduce (average) each parameter gradient across all ranks.

        Call this after ``loss.backward()`` and before ``optimizer.step()``.
        """
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.div_(self.world_size)

class DDP_Bucket(torch.nn.Module):
    """
    Bucket version
    """
    def __init__(self, module: torch.nn.Module,
                 bucket_size_mb: float = 16):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # Make every replica start from the same parameters by broadcasting
        # rank 0's weights (and buffers) to all other ranks.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        # bucket_max_params is the maximum number of parameters that can fit in the bucket
        # we need to find the size of each parameter in bytes and then divide the bucket size by that
        param_dtype = next(self.module.parameters()).dtype
        param_size = param_dtype.itemsize
        bucket_max_params = bucket_size_bytes // param_size
        self.buckets = []
        
        bucket_state = {
            "params": [],
            "need": 0,
            "ready": 0,
            "handle": None,
            "flat": None,
            "grads": None
        }

        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
            param_count = param.numel()
            if bucket_state["need"] > 0 and bucket_state["need"] + param_count > bucket_max_params:
                self.buckets.append(bucket_state)
                bucket_state = {
                    "params": [],
                    "need": 0,
                    "ready": 0,
                    "handle": None,
                    "flat": None,
                    "grads": None
                }
            bucket_state["params"].append(param)
            bucket_state["need"] += param_count
        if bucket_state["need"] > 0:
            self.buckets.append(bucket_state)

        self.register_hooks()
        self._pending = []

    def register_hooks(self):

        def _hook(param: torch.nn.Parameter):
            bucket_idx = param._ddp_bucket_idx
            bucket = self.buckets[bucket_idx]
            bucket["ready"] += param.numel()
            if bucket["ready"] == bucket["need"]:
                grads = [p.grad for p in bucket["params"] if p.grad is not None]
                if len(grads) == 0:
                    bucket["ready"] = 0
                    return
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                bucket["flat"] = flat_grads
                bucket["handle"] = handle
                bucket["grads"] = grads
                bucket["ready"] = 0
                self._pending.append(bucket)
            

        for b_idx, bucket in enumerate(self.buckets):
            for p in bucket["params"]:
                p._ddp_bucket_idx = b_idx
                p.register_post_accumulate_grad_hook(_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket in self._pending:
            bucket["handle"].wait()
            bucket["flat"].div_(self.world_size)
            synced_grads = torch._utils._unflatten_dense_tensors(bucket["flat"], bucket["grads"])
            for i, param in enumerate(bucket["params"]):
                param.grad.copy_(synced_grads[i])
            bucket["handle"] = None
            bucket["flat"] = None
            bucket["grads"] = None
        self._pending.clear()
            


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return DDP(module)
