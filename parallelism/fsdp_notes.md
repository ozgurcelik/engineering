# FSDP notes: async collectives, keepalives, and freeing memory

Notes on some subtle memory/lifetime questions in this FSDP implementation.

## 1. Async collectives and the "keep the input alive" folklore

A collective can be issued asynchronously:

```python
handle = dist.all_gather_into_tensor(out, inp, async_op=True)  # returns immediately
...
handle.wait()   # block until it has actually finished
```

The call returns *before* the communication has necessarily happened. On
GPU/NCCL the op is enqueued on a **separate CUDA stream** and runs concurrently
with the main compute stream; it will read `inp` at some point in the future,
possibly after the Python function that issued it has already returned.

The folklore rule is: **"keep the input tensor alive until `wait()`"**, otherwise
the input's memory might be freed and reused before the collective reads it →
silent corruption.

### Why you don't actually need to do this manually (PyTorch does it)

PyTorch's CUDA caching allocator and NCCL cooperate through **`recordStream`**.
When NCCL launches the collective on its stream, it records that the input's
memory block is in use by that stream. Then, even if the input tensor's Python
refcount drops to zero and the block is "freed", the allocator will **not hand
that block to a new allocation until the NCCL stream has finished the pending
work** (tracked with CUDA events). So the sequence that would corrupt data can't
happen:

1. Last reference to `inp` is dropped.
2. Allocator marks the block *free-but-pending*.
3. A later `torch.empty(...)` will **not** reuse that block yet (still recorded
   against the in-flight NCCL stream).
4. NCCL reads `inp` → still valid.
5. Collective completes → event fires → only now can the block be reused.

Empirically verified: issuing an all-gather from a casted temporary with no
surviving Python reference, then allocating 200 tensors to try to steal its
memory before `wait()`, still produced a bit-exact result.

Aside: with `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, NCCL instead *stashes* the
tensors inside the `Work` handle, which also keeps them alive. Either way,
PyTorch — not the caller — owns the input's lifetime.

The only cases where a manual keepalive genuinely matters:
- **CUDA graph capture** (recordStream semantics don't apply; need static buffers).
- A **custom backend without recordStream** (we only use gloo/NCCL).
- Also, the default overlap windows here (`_reduce_scatter_window_size = 0`,
  `_prefetch_window_size = 1`) mean each collective is `wait()`-ed almost
  immediately, so the input is trivially alive across its whole window anyway.

None of these apply, so `forward/backward_gather_input` and the reduce-scatter
`input_keepalive` are **not required for correctness**. They exist only so the
memory profiler can *see* (account for) the in-flight collective input; they
don't cause that memory to be reserved (NCCL does).

## 2. When is the input memory actually released?

Separate two distinct events:

1. **Reference drop** — refcount hits 0, block returns to the allocator pool.
   Driven purely by Python/C++ references.
2. **Reusable-again** — another allocation can take the block. On CPU this is
   immediate once the ref drops; on **CUDA/NCCL** it *also* needs the
   `recordStream` event to clear, i.e. the collective must have actually finished.

The op running/completing does **not** drop any reference — the collective
reading the input never decrements your refcount. Only a reference going away
does that.

**With the keepalive** (hold the reference until we set it to `None` right after
`wait()`):
- Reference drop happens **after `wait()`** — never earlier, even if the
  collective physically finished long before. The op completing early does *not*
  release it; our explicit `= None` does.

**Without the keepalive:**
- Reference drops **right after issue** (local var goes out of scope).
- Reusable-again: CPU → immediately; NCCL → when the **collective actually
  completes** (recordStream event), which can be *before* we would have hit the
  `= None` line.

So does the keepalive keep memory longer? **Equal-or-longer, never shorter.**
- Comm finishes before we'd clear it (fast comm / lots of intervening compute):
  without keepalive the block is reclaimable at op-completion; with it, not until
  after `wait()`. → keepalive holds it *longer*.
- Comm-bound (`wait()` actually blocks): op-completion ≈ `wait()` ≈ clear point →
  negligible difference.

It can never *save* memory, and can cost a little. Combined with §1, this is why
the keepalives are, at best, memory-neutral observability aids.

### Which "input" even has separate memory?

Only when the input is a distinct, freeable buffer:
- **fp32 all-gather input**: `local_param.detach()` is a *view of the persistent
  master shard* (held forever by the module + optimizer), so it's never freed
  regardless of the keepalive → zero difference.
- **fp16 all-gather input**: `local_param.detach().to(compute_dtype)` is a fresh
  allocation → the only all-gather case where the keepalive governs real memory.
- **reduce-scatter input** at window 0: issue + drain/`wait()` happen in the same
  hook, so the span is tiny and synchronous either way → a wash.

## 3. Why `= None` frees the gather inputs, but `full_param` needs `resize_(0)`

Memory is freed by **refcount**: a `Storage` is returned to the allocator only
when *every* tensor/view pointing at it is gone. Setting one variable to `None`
frees the memory only if it was the *last* reference.

### Gather inputs / grads / RS input are single-owner → `= None` is enough

The casted gather input, `local_param.grad`, and the RS `input_keepalive` each
have exactly one owner:
- The casted gather input is a fresh buffer, consumed only by the collective, and
  it is **detached** — it never enters the layer's autograd graph, so nothing in
  the backward graph saved it.
- After `wait()`, NCCL's `recordStream` hold is released.

So our single stashed reference is the only owner → dropping it (`= None`) takes
the refcount to 0 → freed.

### full_param is multi-owner → dropping references does NOT free it

The `full_param` (all-gathered weight) **participates in the forward matmul**, so
autograd **saves it for backward** (a Linear's backward needs the weight to
compute grad w.r.t. its input). That saved tensor shares the **same storage** as
`full_param`. Now the storage has (at least) two owners:

1. `param_state.full_param` (and the layer attribute) — we control this.
2. Autograd's saved-for-backward reference on the `grad_fn`, held until backward
   runs — we **cannot** reach or drop this.

Therefore the "easy" frees fail:
- `full_param = None` / `setattr(layer, name, local_param)` drops owner #1, but
  owner #2 still holds the storage → memory stays resident until backward. (This
  is the classic FSDP leak: the whole forward weight stack stays alive until
  backward.)
- `full_param.data = torch.empty(0)` just re-points *our* tensor at a new empty
  storage; autograd's saved tensor still points at the **old** storage → no free.

### Why `resize_(0)` works

```python
with torch.no_grad():
    full_param.untyped_storage().resize_(0)
```

We don't try to drop a reference (we can't drop autograd's). Instead we **shrink
the shared storage in place to 0 bytes**. Because owner #1 and owner #2 point at
the *same storage object*, emptying it reclaims the bytes for both at once. The
saved tensor keeps its `[out, in]` metadata (so `AccumulateGrad` still accepts a
full-shaped grad) but its storage now holds nothing — bytes returned immediately,
regardless of how many tensors reference it.

The backward pre-hook (`_regather_full_param_backward`) then resizes that **same**
storage back up and refills it via all-gather, which makes autograd's still-
attached saved tensor valid again.

### The asymmetry in one line

- **Single-owner buffers** (gather input, grad, RS input): you hold the only
  reference → `= None` frees them.
- **Multi-owner buffer** (full_param): autograd also references the same storage
  and you can't drop that reference → refcount-based freeing can't work → empty
  the shared storage in place with `resize_(0)`.

And the reason the input is single-owner is exactly that it's **detached** and
never saved by autograd, whereas the output (`full_param`) is used in the forward
op, so autograd saves it and it becomes multi-owner.
