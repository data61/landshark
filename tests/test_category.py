"""Tests for the category object in the importer code."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from landshark import category
from landshark.basetypes import CategoricalArraySource, CategoricalType


def test_unique_values():
    x = np.array([[1, 2, 2], [1, 2, 3], [1, 1, 2], [1, 1, 1]], dtype=CategoricalType)
    unique_vals, counts = category._unique_values(x)
    true_vals = [np.array([1]), np.array([1, 2]), np.array([1, 2, 3])]
    true_counts = [np.array([4]), np.array([2, 2]), np.array([1, 2, 1])]

    for v, w in zip(unique_vals, true_vals):
        assert np.all(v == w)
    for v, w in zip(counts, true_counts):
        assert np.all(v == w)


def test_category_accumulator():

    missing_value = -1
    acc = category._CategoryAccumulator(missing_value)

    in_data = np.array([-1, 2], dtype=np.int32)
    in_counts = np.array([2, 4], dtype=int)
    in_data_2 = np.array([1, 3], dtype=np.int32)
    in_counts_2 = np.array([1, 1], dtype=int)

    acc.update(in_data, in_counts)
    acc.update(in_data_2, in_counts_2)

    assert set(acc.counts.keys()) == {1, 2, 3}
    assert acc.counts[1] == 1
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
    x = rnd.randint(0, 10, size=(20, 3), dtype=CategoricalType)
    missing_in = -1
    columns = ["1", "2", "3"]
    source = NPCatArraySource(x, missing_in, columns)
    batchsize = 3
    res = category.get_maps(source, batchsize)
    mappings, counts = res.mappings, res.counts
    for m, c, x in zip(mappings, counts, x.T):
        assert set(x) == set(m)
        for m_i, c_i in zip(m, c):
            assert c_i == np.sum(x == m_i)


def test_categorical_transform():

    mappings = [np.array([1, 2, 3]), np.array([1, 2, 4])]
    x = np.array([[2, 2, 3, 1], [4, 1, 1, 2]], dtype=CategoricalType).T
    f = category.CategoryMapper(mappings, missing_value=-1)
    out = f(x)
    ans = np.array([[1, 2], [1, 0], [2, 0], [0, 1]], dtype=CategoricalType)
    assert np.all(out == ans)
