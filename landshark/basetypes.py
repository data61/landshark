"""Baseclass for datatypes. Modron-light basically."""

import logging
from collections import namedtuple

import numpy as np
from typing import Union, Tuple, Optional, List, Sized

log = logging.getLogger(__name__)

# Definitions of numerical types used in the project.
OrdinalType = np.float32
CategoricalType = np.int32
NumericalType = Union[np.float32, np.int32]
MissingType = Optional[NumericalType]
CoordinateType = np.float64

FeatureValues = namedtuple("FeatureValues",
                           ["ordinal", "categorical", "coordinates"])


class FixedSlice:
    """
    Slice object that requires a start and end point.

    This is mainly for typing reasons, in a normal slice start and stop
    can be None. This is not allowed in this class.

    Parameters
    ----------
    start : int
        Start of the slice (inclusive).
    stop : int
        End of the slice (exclusive).

    """

    def __init__(self, start: int, stop: int) -> None:
        """Initialise the object."""
        assert start >= 0
        assert stop >= start
        self.start = start
        self.stop = stop

    @property
    def as_slice(self) -> slice:
        """Convert to a python native slice."""
        s = slice(self.start, self.stop)
        return s


class ArraySource(Sized):
    """Abstract UniData interface."""

    def __init__(self) -> None:
        """Baseclass for data backends."""
        self._shape = (0, 0)
        self._native = 0
        self._dtype = OrdinalType
        self._missing: List[MissingType] = []
        self._columns: List[str] = []

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
        """
        Get the native access size (in rows) of the data source.

        Returns
        -------
        rows: int

        """
        return self._native

    @property
    def missing(self) -> List[MissingType]:
        """
        Get the special values that show missing data for each feature.

        Features are indexed by the LAST axis.

        Returns
        -------
        m : List[MissingType]
            A list entry for each feature, with either the missing value
            or None to indicate there is no missing value defined.

        """
        return self._missing

    @property
    def columns(self) -> List[str]:
        """
        Get text descriptors of each (column) feature.

        Returns
        -------
        l : List[str]
            A list of feature/column names. Should be the same length
            as the last dimension of the data.

        """
        return self._columns

    def slice(self, start: int, end: int) -> np.ndarray:
        """
        Get a slice from the array along the first dimension.

        Parameters
        ----------
        start : int
            The start of the slice (inclusive)
        end : int
            The end of the slice (exclusive)

        Returns
        -------
        array : np.ndarray
            The values of the slice. First dimension will have length
            end - start.

        """
        return self._arrayslice(start, end)

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        """Perform the array slice. This gets overridden by children."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of rows (1st dimension)."""
        return self._shape[0]


class OrdinalArraySource(ArraySource):
    """Array source for Ordinal data."""

    _dtype = OrdinalType


class CategoricalArraySource(ArraySource):
    """Array source for categorical data."""

    _dtype = CategoricalType


class CoordinateArraySource(ArraySource):
    """Array source for coordinate data."""

    _dtype = CoordinateType
