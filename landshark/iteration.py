import numpy as np
import itertools

def batch(it, batchsize, total_size):
    while True:
        batch = list(itertools.islice(it, batchsize))
        if not batch:
            return
        yield batch


def batch_slices(batchsize, total_size):
    n = total_size // batchsize
    ret = [(i * batchsize, (i + 1) * batchsize) for i in range(n)]
    if total_size % batchsize != 0:
        ret.append((n * batchsize, total_size))
    return iter(ret)

def with_slices(it):
    """Needs iterator over ndarrays"""
    start_idx = 0
    for d in it:
        end_idx = start_idx + d.shape[0]
        yield (start_idx, end_idx), d
        start_idx = end_idx
