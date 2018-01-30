"""Tests for the basetypes module."""

import numpy as np

from landshark import basetypes


def test_fixedslice():
    x = basetypes.FixedSlice(4, 8)
    assert x.start == 4
    assert x.stop == 8
    assert x.as_slice == slice(4, 8)


def test_featurevalues(mocker):
    ord_src = mocker.Mock()
    cat_src = mocker.Mock()
    coord_src = mocker.Mock()
    x = basetypes.FeatureValues(ord_src, cat_src, coord_src)
    assert x.ordinal is ord_src
    assert x.categorical is cat_src
    assert x.coordinates is coord_src


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
    assert basetypes.OrdinalArraySource._dtype == basetypes.OrdinalType
    assert basetypes.CoordinateArraySource._dtype == basetypes.CoordinateType
