import logging
import numpy as np
from collections import OrderedDict
from landshark import iteration

log = logging.getLogger(__name__)

class CategoryPreprocessor:

    def __init__(self, ncols):
        self.ncols = ncols

    def __call__(self, values):
        x = values.categorical
        x = x.reshape((-1), self.ncols)
        unique_vals = [np.unique(c) for c in x.T]
        counts = [np.array([np.sum(c == v) for v in uv])
                  for uv, c in zip(unique_vals, x.T)]
        return unique_vals, counts


class CategoryMapper:
    def __init__(self):
        self.counts = OrderedDict()

    def update(self, values, counts):
        for v, c in zip(values, counts):
            if v in self.counts:
                self.counts[v] += c
            else:
                self.counts[v] = c

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


def get_categories(source, batchsize, pool):
    array_src = source.categorical
    n_rows = array_src.shape[0]
    n_features = array_src.shape[-1]
    missing_values = array_src.missing
    mappers = [CategoryMapper() for _ in range(n_features)]

    it = iteration.batch_slices(batchsize, n_rows)
    f = CategoryPreprocessor(n_features)
    data_it = ((source.slice(start, end)) for start, end in it)
    out_it = pool.imap(f, data_it)
    for mapper, m in zip(mappers, missing_values):
        if m is not None:
            mapper.update([m], [0])

    log.info("Computing unique values in categorical features:")
    for unique_vals, counts in out_it:
        for mapper, u, c in zip(mappers, unique_vals, counts):
            mapper.update(u, c)

    count_dicts = [m.counts for m in mappers]
    mappings = [np.array(list(c.keys()), dtype=np.int32) for c in count_dicts]
    counts = [np.array(list(c.values()), dtype=np.int64) for c in count_dicts]
    return mappings, counts

