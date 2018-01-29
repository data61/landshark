"""Tests for the category object in the importer code."""
from multiprocessing import Pool

import numpy as np
from landshark import category
from landshark.basetypes import CategoricalValues, CategoricalDataSource, CategoricalArraySource, CategoricalType



def test_unique_values():
    in_data = np.array([[1, 2, 2], [1, 2, 3],
                        [1, 1, 2], [1, 1, 1]])
    x = CategoricalValues(in_data)
    unique_vals, counts = category._unique_values(x)
    true_vals = [np.array([1]), np.array([1, 2]), np.array([1, 2, 3])]
    true_counts = [np.array([4]), np.array([2, 2]), np.array([1, 2, 1])]

    for v, w in zip(unique_vals, true_vals):
        assert np.all(v == w)
    for v, w in zip(counts, true_counts):
        assert np.all(v == w)


def test_category_accumulator():

    acc = category._CategoryAccumulator()

    in_data = np.array([1, 2], dtype=np.int32)
    in_counts = np.array([2, 4], dtype=int)
    in_data_2 = np.array([1, 3], dtype=np.int32)
    in_counts_2 = np.array([1, 1], dtype=int)

    acc.update(in_data, in_counts)
    acc.update(in_data_2, in_counts_2)

    assert acc.counts[1] == 3
    assert acc.counts[2] == 4
    assert acc.counts[3] == 1


class NPCatArraySource(CategoricalArraySource):
    def __init__(self, x, missing, columns):
        self._shape = x.shape
        self._native = 1
        self._missing = missing
        self._columns = columns
        self._data = x

    def _arrayslice(self, start, stop):
        return self._data[start:stop]


def test_get_categories(mocker):
    rnd = np.random.RandomState(seed=666)
    x = rnd.randint(0, 10, size=(20, 3))
    missing_in = [None, 0, 1]
    columns = ["1", "2", "3"]
    source = CategoricalDataSource(NPCatArraySource(x, missing_in, columns))
    pool = Pool(2)
    batchsize = 3
    res = category.get_categories(source, batchsize, pool)
    mappings, counts, missing = res.mappings, res.counts, res.missing
    assert missing == [None, 0, 0]
    for m, c, x in zip(mappings, counts, x.T):
        assert set(x) == set(m)
        for m_i, c_i in zip(m, c):
            assert c_i == np.sum(x == m_i)


def test_categorical_transform():

    mappings = [np.array([1, 2, 3]), np.array([4, 1, 2])]
    x = np.array([[2, 2, 3, 1], [4, 1, 1, 2]], dtype=CategoricalType).T
    values = CategoricalValues(x)
    f = category.CategoricalOutputTransform(mappings)
    out = f(values)
    ans = np.array([[1, 0],
                    [1, 1],
                    [2, 1],
                    [0, 2]], dtype=CategoricalType)
    assert np.all(out == ans)

# def test_category_obj_missing():
#     missing_values = [None, 11]
#     cat = category._Categories(missing_values)
#     assert cat.missing_values == [None, 0]

# def test_category_obj():

#     missing_values = [None, 11]
#     data = np.array([[0, 10],
#                      [1, 11],
#                      [0, 10],
#                      [4, 14],
#                      [5, 15],
#                      [1, 11],
#                      [6, 16]])

#     data_list = [data[0:2], data[2:5], data[5:]]
#     new_list = []
#     cat = category._Categories(missing_values)
#     for d in data_list:
#         new_list.append(cat.update(d))
#     new_data = np.concatenate(new_list, axis=0)
#     answer = np.array([[0, 1],
#                        [1, 0],
#                        [0, 1],
#                        [2, 2],
#                        [3, 3],
#                        [1, 0],
#                        [4, 4]])
#     assert np.all(new_data == answer)
