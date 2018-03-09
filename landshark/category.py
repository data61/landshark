"""Operations to support categorical data."""

import logging
from collections import OrderedDict, namedtuple

from tqdm import tqdm
import numpy as np
from typing import Tuple, List

from landshark import iteration
from landshark.basetypes import CategoricalType

log = logging.getLogger(__name__)

CategoryInfo = namedtuple("CategoryInfo", ["mappings", "counts", "missing"])


def unique_values(x: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    x = x.reshape((-1), x.shape[-1])
    unique_vals, counts = zip(*[np.unique(c, return_counts=True)
                                for c in x.T])
    return unique_vals, counts


class _CategoryAccumulator:
    """Class for accumulating categorical values and their counts."""

    def __init__(self) -> None:
        """Initialise the object."""
        self.counts: OrderedDict = OrderedDict()

    def update(self, values: np.ndarray, counts: np.ndarray) -> None:
        """Add a new set of values from a batch."""
        assert values.ndim == 1
        assert counts.ndim == 1
        assert values.shape == counts.shape
        assert counts.dtype == int
        assert np.all(counts >= 0)
        for v, c in zip(values, counts):
            if v in self.counts:
                self.counts[v] += c
            else:
                self.counts[v] = c


def get_maps(src, batchsize: int, n_workers: int) -> CategoryInfo:
    """
    Extract the unique categorical variables and their counts.

    TODO
    """
    n_rows = src.shape[0]
    n_features = src.shape[-1]
    missing_value = src.missing
    accums = [_CategoryAccumulator() for _ in range(n_features)]
    # Add the missing values initially as zeros
    if missing_value is not None:
        for a in accums:
            a.update(np.array([missing_value]), np.array([0], dtype=int))

    with tqdm(total=n_rows) as pbar:
        with src:
            for s in iteration.batch_slices(batchsize, n_rows):
                x = src(s)
                unique, counts = unique_values(x)
                for a, u, c in zip(accums, unique, counts):
                    a.update(u, c)
                pbar.update(x.shape[0])

    missing = CategoricalType(0) if missing_value is not None else None
    count_dicts = [m.counts for m in accums]
    mappings = [np.array(list(c.keys())) for c in count_dicts]
    counts = [np.array(list(c.values()), dtype=np.int64) for c in count_dicts]
    result = CategoryInfo(mappings=mappings, counts=counts, missing=missing)
    return result


class CategoryMapper:
    def __init__(self, mappings: List[np.ndarray]) -> None:
        self._mappings = mappings

    def __call__(self, x: np.ndarray) -> np.ndarray:
        buf = np.empty_like(x)
        for ch, cat in enumerate(self._mappings):
            flat = np.hstack((cat, x[..., ch].ravel()))
            _, remap = np.unique(flat, return_inverse=True)
            buf[..., ch] = remap[len(cat):].reshape(x.shape[:-1])
        return buf
