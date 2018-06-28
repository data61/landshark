"""Operations to support categorical data."""

import logging
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
from typing import Tuple, List, NamedTuple, Optional, Callable

from landshark import iteration
from landshark.basetypes import CategoricalType, CategoricalArraySource, Worker

log = logging.getLogger(__name__)

class CategoryInfo(NamedTuple):
    mappings: List[np.ndarray]
    counts: List[np.ndarray]


def unique_values(x: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    x = x.reshape((-1), x.shape[-1])
    unique_vals, counts = zip(*[np.unique(c, return_counts=True)
                                for c in x.T])
    return unique_vals, counts


class _CategoryAccumulator:
    """Class for accumulating categorical values and their counts."""

    def __init__(self, missing_value: CategoricalType) -> None:
        """Initialise the object."""
        self.counts: OrderedDict = OrderedDict()
        self.missing = missing_value

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
        # Dont include the missing value
        if self.missing in self.counts:
            self.counts.pop(self.missing)


def get_maps(src: CategoricalArraySource, batchrows: int) -> CategoryInfo:
    """
    Extract the unique categorical variables and their counts.

    TODO
    """
    n_rows = src.shape[0]
    n_features = src.shape[-1]
    missing_value = src.missing
    accums = [_CategoryAccumulator(missing_value) for _ in range(n_features)]

    if missing_value is not None and missing_value > 0:
        raise ValueError("Missing value must be negative")

    with tqdm(total=n_rows) as pbar:
        with src:
            for s in iteration.batch_slices(batchrows, n_rows):
                x = src(s)
                unique, counts = unique_values(x)
                for a, u, c in zip(accums, unique, counts):
                    a.update(u, c)
                pbar.update(x.shape[0])

    count_dicts = [m.counts for m in accums]
    unsorted_mappings = [np.array(list(c.keys())) for c in count_dicts]
    unsorted_counts = [np.array(list(c.values()), dtype=np.int64)
                       for c in count_dicts]
    sortings = [np.argsort(m, kind="mergesort") for m in unsorted_mappings]
    mappings = [m[s] for m, s in zip(unsorted_mappings, sortings)]
    counts = [c[s] for c, s in zip(unsorted_counts, sortings)]
    result = CategoryInfo(mappings=mappings, counts=counts)
    return result


class CategoryMapper(Worker):
    def __init__(self, mappings: List[np.ndarray],
                 missing_value: Optional[int]) -> None:
        for m in mappings:
            is_sorted = np.all(m[:-1] <= m[1:])
            assert is_sorted
        self._mappings = mappings
        self._missing = missing_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        fill = self._missing if self._missing is not None else 0
        x_new = np.empty_like(x)
        for i, cats in enumerate(self._mappings):
            x_i = x[..., i].ravel()
            mask = x_i != self._missing if self._missing \
                else np.ones_like(x_i, dtype=bool)
            x_i_valid = x_i[mask].flatten()
            flat = np.hstack((cats, x_i_valid))
            actual_cat, remap = np.unique(flat, return_inverse=True)
            x_i_new_valid = remap[len(cats):]
            x_i_new = np.full_like(x_i, fill)
            x_i_new[mask] = x_i_new_valid
            x_new[..., i] = x_i_new.reshape(x[..., i].shape)
            assert np.all(actual_cat == cats)
        return x_new
