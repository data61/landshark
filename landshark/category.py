"""Operations to support categorical data."""

import logging
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from typing import Tuple, List


from landshark.basetypes import CategoricalValues, CategoricalDataSource
from landshark import iteration

log = logging.getLogger(__name__)


class _CategoryPreprocessor:
    """Stateful function for returning unique values."""

    def __init__(self, ncols: int) -> None:
        """Initialise the object."""
        assert ncols > 0
        self.ncols = ncols

    def __call__(self, values: CategoricalValues) \
            -> Tuple[List[np.ndarray], List[int]]:
        """Get the unique values and their counts from the input."""
        x = values.categorical
        assert x.shape[-1] == self.ncols
        x = x.reshape((-1), self.ncols)
        unique_vals = [np.unique(c) for c in x.T]
        counts = [np.array([np.sum(c == v) for v in uv])
                  for uv, c in zip(unique_vals, x.T)]
        return unique_vals, counts


class _CategoryAccumulator:
    """Class for accumulating categorical values and their counts."""

    def __init__(self) -> None:
        """Initialise the object."""
        self.counts: OrderedDict = OrderedDict()

    def update(self, values: List[np.ndarray], counts: List[int]) -> None:
        """Add a new set of values from a batch."""
        for v, c in zip(values, counts):
            if v in self.counts:
                self.counts[v] += c
            else:
                self.counts[v] = c


def get_categories(source: CategoricalDataSource,
                   batchsize: int,
                   pool: Pool) -> None:
    """
    Extract the unique categorical variables and their counts.

    Parameters
    ----------
    source : CategoricalDataSource
        The data source to examine.
    batchsize : int
        The number of rows to examine in one iteration (by 1 proc)
    pool : multiprocessing.Pool
        The pool of processes over which to distribute the work.

    Returns
    -------
    mappings : List[np.ndarray]
        The mapping of unique values for each feature.
    counts : List[np.ndarray]
        The counts of unique values for each feature.


    """
    array_src = source.categorical
    n_rows = array_src.shape[0]
    n_features = array_src.shape[-1]
    missing_values = array_src.missing
    accums = [_CategoryAccumulator() for _ in range(n_features)]

    it = iteration.batch_slices(batchsize, n_rows)
    f = _CategoryPreprocessor(n_features)
    data_it = ((source.slice(start, end)) for start, end in it)
    out_it = pool.imap(f, data_it)
    for acc, m in zip(accums, missing_values):
        if m is not None:
            acc.update([m], [0])

    log.info("Computing unique values in categorical features:")
    for unique_vals, counts in out_it:
        for mapper, u, c in zip(accums, unique_vals, counts):
            mapper.update(u, c)

    count_dicts = [m.counts for m in accums]
    mappings = [np.array(list(c.keys()), dtype=np.int32) for c in count_dicts]
    counts = [np.array(list(c.values()), dtype=np.int64) for c in count_dicts]
    return mappings, counts


class CategoricalOutputTransform:
    def __init__(self, mappings):
        self.mappings = mappings

    def __call__(self, values):
        x = values.categorical
        new_array = np.zeros_like(x, dtype=x.dtype)
        for col_idx, m in enumerate(self.mappings):
            old_col = x[..., col_idx]
            new_col = new_array[..., col_idx]
            for i, v in enumerate(m):
                indices = old_col == v
                new_col[indices] = i
                return new_array


