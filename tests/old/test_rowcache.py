"""Tests for the rowcache submodule."""

import numpy as np
import pytest

from landshark import rowcache


def test_rowcache_init() -> None:
    """Check the rowcache initializes properly."""
    harray = np.arange(56).reshape((7, 8))
    rows_per_block = 3
    n_blocks = 3

    cache = rowcache.RowCache(harray, rows_per_block, n_blocks)

    assert cache._harray is harray
    assert cache.total_rows == harray.shape[0]
    assert cache.rows_per_block == rows_per_block

    assert cache.block_slices == [slice(0, 3), slice(3, 6), slice(6, 7)]
    assert set(cache._d.keys()) == {2, 1, 0}
    for i, s in enumerate(cache.block_slices):
        assert np.all(cache._d[i] == harray[s])


def test_rowcache_init_single() -> None:
    """Check the case where there's only one block."""
    harray = np.arange(56).reshape((7, 8))
    rows_per_block = 10
    n_blocks = 3
    # This should be impossible as there is only 1 block
    cache = rowcache.RowCache(harray, rows_per_block, n_blocks)
    assert cache.nblocks == 1


def test_rowcache_evict() -> None:
    """Check the LRU cache works properly."""
    harray = np.arange(56).reshape((7, 8))
    rows_per_block = 2
    n_blocks = 2
    # This should be impossible as there is only 1 block
    cache = rowcache.RowCache(harray, rows_per_block, n_blocks)
    assert set(cache._d.keys()) == {0, 1}
    cache(6, slice(0, 8))
    assert set(cache._d.keys()) == {1, 3}
    cache(0, slice(0, 8))
    assert set(cache._d.keys()) == {3, 0}


@pytest.mark.parametrize("shape", np.random.randint(1, 30, size=(100, 2)))
def test_rowcache_get(shape: np.ndarray) -> None:
    """Check the get functionality works as expected."""
    height = shape[0]
    width = shape[1]
    rows_per_block = np.random.randint(1, height + 1)
    n_blocks = np.random.randint(1, height + 1)
    harray = np.arange(width * height).reshape((height, width))
    cache = rowcache.RowCache(harray, rows_per_block, n_blocks)
    yval = np.random.randint(height)
    xstart = np.random.randint(width)
    xstop = min(xstart + np.random.randint(width), width - 1)
    xslice = slice(xstart, xstop)
    ans = harray[yval, xslice]
    res = cache(yval, xslice)
    assert np.all(ans == res)
