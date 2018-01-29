"""Tests for the basetypes module."""

import numpy as np

from landshark import basetypes


def test_fixedslice():
    x = basetypes.FixedSlice(4, 8)
    assert x.start == 4
    assert x.stop == 8
    assert x.as_slice == slice(4, 8)


def test_categorical_values():
    x = np.ones((3, 2), dtype=basetypes.CategoricalType)
    v = basetypes.CategoricalValues(x)
    assert v.categorical is x


def test_ordinal_values():
    x = np.ones((3, 2), dtype=basetypes.OrdinalType)
    v = basetypes.OrdinalValues(x)
    assert v.ordinal is x


def test_mixed_values():
    x_ord = np.ones((3, 2), dtype=basetypes.OrdinalType)
    x_cat = np.ones((3, 4), dtype=basetypes.CategoricalType)
    v = basetypes.MixedValues(x_ord, x_cat)
    assert v.ordinal is x_ord
    assert v.categorical is x_cat

def test_categorical_coords():
    x = np.ones((3, 2), dtype=basetypes.CategoricalType)
    c = np.zeros((3, 2), dtype=basetypes.CoordinateType)
    v = basetypes.CategoricalCoords(x, c)
    assert v.coords is c
    assert v.categorical is x


def test_ordinal_coords():
    x = np.ones((3, 2), dtype=basetypes.OrdinalType)
    c = np.zeros((3, 2), dtype=basetypes.CoordinateType)
    v = basetypes.OrdinalCoords(x, c)
    assert v.ordinal is x
    assert v.coords is c


def test_mixed_coords():
    x_ord = np.ones((3, 2), dtype=basetypes.OrdinalType)
    x_cat = np.ones((3, 4), dtype=basetypes.CategoricalType)
    c = np.zeros((3, 2), dtype=basetypes.CoordinateType)
    v = basetypes.MixedCoords(x_ord, x_cat, c)
    assert v.ordinal is x_ord
    assert v.categorical is x_cat
    assert v.coords is c


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


def get_sources(m):
    m_ord_src = m.Mock()
    m_ord_src.shape = (3, 5)
    m_ord_src._arrayslice = m.Mock()
    m_ord_src.__len__ = m.Mock()
    m_ord_src.__len__.return_value = 3
    m_ord_src._arrayslice.return_value = np.ones((3, 5),
                                                 dtype=basetypes.OrdinalType)
    m_cat_src = m.Mock()
    m_cat_src.shape = (3, 4)
    m_cat_src.__len__ = m.Mock()
    m_cat_src.__len__.return_value = 3
    m_cat_src._arrayslice = m.Mock()
    m_cat_src._arrayslice.return_value =  \
        np.ones((3, 4), dtype=basetypes.CategoricalType)

    m_coord_src = m.Mock()
    m_coord_src.shape = (3, 2)
    m_coord_src.__len__ = m.Mock()
    m_coord_src.__len__.return_value = 3
    m_coord_src._arrayslice = m.Mock()
    m_coord_src._arrayslice.return_value =  \
        np.ones((3, 2), dtype=basetypes.CoordinateType)
    return m_ord_src, m_cat_src, m_coord_src


def test_ordinal_datasource(mocker):

    m_src, _, _ = get_sources(mocker)
    v = basetypes.OrdinalDataSource(m_src)
    assert v.slice(1, 2).ordinal is m_src._arrayslice.return_value
    assert v.ordinal is m_src
    assert len(v) == 3


def test_categorical_datasource(mocker):
    _, m_src, _ = get_sources(mocker)
    v = basetypes.CategoricalDataSource(m_src)
    assert v.slice(1, 2).categorical is m_src._arrayslice.return_value
    assert v.categorical is m_src
    assert len(v) == 3


def test_mixed_datasource(mocker):
    m_ord_src, m_cat_src, _ = get_sources(mocker)
    v = basetypes.MixedDataSource(m_ord_src, m_cat_src)
    assert v.slice(1, 2).categorical is m_cat_src._arrayslice.return_value
    assert v.slice(1, 2).ordinal is m_ord_src._arrayslice.return_value
    assert v.categorical is m_cat_src
    assert v.ordinal is m_ord_src
    assert len(v) == 3

def test_ordinal_coordource(mocker):
    m_src, _, m_coords = get_sources(mocker)
    v = basetypes.OrdinalCoordSource(m_src, m_coords)
    assert v.slice(1, 2).ordinal is m_src._arrayslice.return_value
    assert v.slice(1, 2).coords is m_coords._arrayslice.return_value
    assert v.ordinal is m_src
    assert v.coordinates is m_coords
    assert len(v) == 3


def test_categorical_coordsource(mocker):
    _, m_src, m_coords = get_sources(mocker)
    v = basetypes.CategoricalCoordSource(m_src, m_coords)
    assert v.slice(1, 2).categorical is m_src._arrayslice.return_value
    assert v.slice(1, 2).coords is m_coords._arrayslice.return_value
    assert v.categorical is m_src
    assert v.coordinates is m_coords
    assert len(v) == 3

def test_mixed_coordsource(mocker):

    m_ord_src, m_cat_src, m_coords = get_sources(mocker)
    v = basetypes.MixedCoordSource(m_ord_src, m_cat_src, m_coords)
    assert v.slice(1, 2).categorical is m_cat_src._arrayslice.return_value
    assert v.slice(1, 2).ordinal is m_ord_src._arrayslice.return_value
    assert v.slice(1, 2).coords is m_coords._arrayslice.return_value
    assert v.categorical is m_cat_src
    assert v.ordinal is m_ord_src
    assert v.coordinates is m_coords
    assert len(v) == 3
