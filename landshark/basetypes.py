"""Baseclass for datatypes. Modron-light basically."""

import logging

import numpy as np
from typing import Union, Tuple

log = logging.getLogger(__name__)

NumericalType = Union[np.integer, np.floating]
OrdinalType = np.float32
CategoricalType = np.int32
CoordinateType = np.float64


class CategoricalValues:
    def __init__(self, array: np.ndarray):
        self.categorical = array


class OrdinalValues:
    def __init__(self, array: np.ndarray):
        self.ordinal = array


class MixedValues(CategoricalValues, OrdinalValues):
    def __init__(self, ord_array, cat_array):
        assert len(ord_array) == len(cat_array)
        OrdinalValues.__init__(self, ord_array)
        CategoricalValues.__init__(self, cat_array),


class CategoricalCoords(CategoricalValues):
    def __init__(self, array: np.ndarray, coords: np.ndarray):
        super().__init__(array)
        self.coords = coords


class OrdinalCoords(OrdinalValues):
    def __init__(self, array: np.ndarray, coords: np.ndarray):
        super().__init__(array)
        self.coords = coords


class MixedCoords(MixedValues):
    def __init__(self, ord_array: np.ndarray, cat_array: np.ndarray,
                 coords: np.ndarray):
        super().__init__(ord_array, cat_array)
        self.coords = coords


Values = Union[CategoricalValues, OrdinalValues, MixedValues]


class DataSource:
    def __init__(self) -> None:
        """Stub init for data type."""
        self._n = 0

    def slice(self, start: int, end: int) -> Values:
        """Implement slice..."""
        return self._slice(start, end)

    def _slice(self, start: int, end: int) -> Values:
        raise NotImplementedError

    def __len__(self) -> int:
        return self._n


class ArraySource:
    """Abstract UniData interface."""

    def __init__(self) -> None:
        """Baseclass for data backends."""
        self._shape = (0, 0)
        self._native = 0
        self._dtype = OrdinalType
        self._missing = []
        self._columns = []

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape tuple of Dataset.

        Returns
        -------
        shape: Tuple[int, ...]

        """
        return self._shape

    @property
    def dtype(self) -> NumericalType:
        """Get the datatype of the underlying stream.

        Returns:
        --------
        type: type

        """
        return self._dtype

    @property
    def native(self) -> int:
        """Get the native access size (in rows) of the data source.

        Returns
        -------
        rows: int

        """
        return self._native

    @property
    def missing(self):
        """Missing values."""
        return self._missing

    @property
    def columns(self):
        return self._columns


    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        raise NotImplementedError


class OrdinalArraySource:
    _dtype = OrdinalType


class CategoricalArraySource:
    _dtype = CategoricalType


class OrdinalDataSource(DataSource):

    def __init__(self, source: OrdinalArraySource) -> None:
        self.ordinal = source
        self._n = self.ordinal.shape[0]

    def _slice(self, start: int, end: int) -> OrdinalValues:
        result = OrdinalValues(self.ordinal._arrayslice(start, end))
        return result

class OrdinalCoordSource(OrdinalDataSource):
    def __init__(self, ord_source: OrdinalArraySource,
                 coord_source: OrdinalArraySource) -> None:
        super().__init__(ord_source)
        self.coordinates = coord_source

    def _slice(self, start: int, end: int) -> OrdinalCoords:
        result = OrdinalCoords(self.ordinal._arrayslice(start, end),
                               self.coordinates._arrayslice(start, end))
        return result

class CategoricalDataSource(DataSource):

    def __init__(self, source: CategoricalArraySource) -> None:
        self.categorical = source
        self._n = self.categorical.shape[0]

    def _slice(self, start: int, end: int) -> CategoricalValues:
        result = CategoricalValues(self.categorical._arrayslice(start, end))
        return result

class CategoricalCoordSource(CategoricalDataSource):
    def __init__(self, cat_source: CategoricalArraySource,
                 coord_source: OrdinalArraySource) -> None:
        super().__init__(cat_source)
        self.coordinates = coord_source

    def _slice(self, start: int, end: int) -> CategoricalCoords:
        result = CategoricalCoords(self.categorical._arrayslice(start, end),
                                   self.coordinates._arrayslice(start, end))
        return result


class MixedDataSource(DataSource):

    def __init__(self, ordinal_source: OrdinalArraySource,
                 categorical_source: CategoricalArraySource,):
        self.ordinal = ordinal_source
        self.categorical = categorical_source
        assert ordinal_source.shape[0] == categorical_source.shape[0]
        self._n = ordinal_source.shape[0]

    def _slice(self, start: int, stop: int):
        result = MixedValues(self.ordinal._arrayslice(start, stop),
                             self.categorical._arrayslice(start, stop))
        return result
