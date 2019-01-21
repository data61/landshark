"""Test iteration module."""

import numpy as np
import pytest

from landshark.iteration import batch, batch_slices, with_slices

batch_params = [
    (10, 5),
    (123456, 79)
]


@pytest.mark.parametrize("N,B", batch_params)
def test_batch(N, B):
    it = iter(range(N))
    ixs = list(batch(it, B))
    bs = [len(b) for b in ixs]
    assert bs == [B] * (N // B) + [] if N % B == 0 else [N % B]
    ixs_flat = [i for b in ixs for i in b]
    assert ixs_flat == list(range(N))


@pytest.mark.parametrize("N,B", batch_params)
def test_batch_slices(N, B):
    ixs = list(batch_slices(B, N))
    start, stop = tuple(zip(*[(s.start, s.stop) for s in ixs]))
    assert start == tuple(range(0, N, B))
    assert stop == tuple(list(range(B, N, B)) + [N])


@pytest.mark.parametrize("N", [1, 10])
def test_with_slices(N):
    n_rows = np.random.randint(1, 10, N)
    n_cols = 3
    arrays = [np.random.rand(r, n_cols) for r in n_rows]
    slices, arrays2 = tuple(zip(*with_slices(arrays)))
    assert len(arrays) == len(arrays2)
    for a, a2 in zip(arrays, arrays2):
        np.testing.assert_array_equal(a, a2)
    start, stop = tuple(zip(*[(s.start, s.stop) for s in slices]))
    n_rows_sum = np.insert(np.cumsum(n_rows), 0, 0)
    assert start == tuple(n_rows_sum[:-1])
    assert stop == tuple(n_rows_sum[1:])
