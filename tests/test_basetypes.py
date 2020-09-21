"""Tests for the basetypes module."""

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

from landshark import basetypes


def test_fixedslice():
    x = basetypes.FixedSlice(4, 8)
    assert x.start == 4
    assert x.stop == 8


def test_featurevalues(mocker):
    con_src = mocker.Mock()
    cat_src = mocker.Mock()
    x = basetypes.FeatureValues(con_src, cat_src)
    assert x.continuous is con_src
    assert x.categorical is cat_src


class NpyCatArraySource(basetypes.CategoricalArraySource):
    def __init__(self, x, missing, columns):
        self._shape = x.shape
        self._native = 1
        self._missing = missing
        self._columns = columns
        self._data = x

    def _arrayslice(self, start, stop):
        return self._data[start:stop]


def test_array_source():
    x = np.ones((3, 2), dtype=basetypes.CategoricalType)
    missing = [basetypes.CategoricalType(1), None]
    columns = ["1", "2"]
    s = NpyCatArraySource(x, missing, columns)
    assert s.columns is columns
    assert s.dtype == basetypes.CategoricalType
    assert s.missing is missing
    assert s.native == 1
    assert s.shape == x.shape
    assert basetypes.ContinuousArraySource._dtype == basetypes.ContinuousType
    assert basetypes.CoordinateArraySource._dtype == basetypes.CoordinateType
