import numpy as np


def _update_maps(map_dict, value_set, data, max_categories=5000):
    unique_vals = np.unique(data)
    new_values = set(unique_vals).difference(value_set)
    nstart = len(value_set)
    nstop = nstart + len(new_values) + 1
    new_indices = range(nstart, nstop)
    map_dict.update(zip(new_values, new_indices))
    value_set.update(new_values)
    assert(len(value_set) < max_categories)

def _transform_cats(map_dict, old_array, new_array):
    for k, v in map_dict.items():
        new_array[old_array == k] = v

def _transform_missing(missing_values):
    result = [(np.int32(0) if k is not None else None) for k in missing_values]
    return result

def _map_to_list(map_dict):
    result = [i[0] for i in sorted(map_dict.items(), key=lambda x: x[1])]
    return result


class _Categories:
    """Class that gets the number of categories for features."""
    def __init__(self, missing_values) -> None:
        self._missing_values = _transform_missing(missing_values)
        n_features = len(missing_values)
        self._values = [set() for _ in range(n_features)]
        self._maps = [dict() for _ in range(n_features)]
        for i, k in enumerate(missing_values):
            if k is not None:
                self._values[i].add(k)
                self._maps[i][k] = np.int32(0)

    def update(self, array: np.ndarray):
        new_array = np.zeros_like(array, dtype=np.int32)
        for i, data in enumerate(array.T):
            _update_maps(self._maps[i], self._values[i], data)
            _transform_cats(self._maps[i], array[..., i], new_array[..., i])
        return new_array

    @property
    def missing_values(self):
        return self._missing_values

    @property
    def maps(self):
        map_list = [_map_to_list(k) for k in self._maps]
        return map_list

