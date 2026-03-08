"""Wrappers that set torch print precision and fix data seeding for verification."""
import torch
torch.set_printoptions(precision=10)

import parallelism

# Monkey-patch generate_sample_data to use a fixed seed so baseline and
# parallel runs produce the same data.
_orig_generate = parallelism.generate_sample_data
def _seeded_generate():
    torch.manual_seed(42)
    return _orig_generate()
parallelism.generate_sample_data = _seeded_generate

from parallelism import pipeline_parallelism_main, tensor_parallelism_main

def pipeline_wrapper(rank, *args):
    torch.set_printoptions(precision=10)
    pipeline_parallelism_main(rank, *args)

def tensor_wrapper(rank, *args):
    torch.set_printoptions(precision=10)
    tensor_parallelism_main(rank, *args)
