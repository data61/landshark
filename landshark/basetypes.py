"""Baseclass for datatypes. Modron-light basically."""

import logging
from collections import namedtuple

import numpy as np
from typing import Union, Tuple, Optional, List, Sized, NamedTuple, Any, Dict

log = logging.getLogger(__name__)

# Definitions of numerical types used in the project.
OrdinalType = np.float32
CategoricalType = np.int32
NumericalType = Union[np.float32, np.int32]
MissingType = Optional[NumericalType]
CoordinateType = np.float64
DataType = Union[OrdinalType, CategoricalType, CoordinateType]

FeatureValues = namedtuple("FeatureValues",
                           ["ordinal", "categorical"])


class FixedSlice(NamedTuple):
    """simpler slice."""

    start: int
    stop: int

class ArraySource(Sized):
    """Abstract UniData interface."""

    def __init__(self) -> None:
        """Baseclass for data backends."""
        self._shape = (0, 0)
        self._native = 0
        self._dtype = OrdinalType
        self._missing: MissingType = None
        self._columns: List[str] = []
        self._open = False

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
    def missing(self) -> MissingType:
        """
        Get the special value that show missing data for the features.

        Returns
        -------
        m : MissingType
            The value to indicate missingness or none if there is no
            missing values
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

    def __call__(self, s: FixedSlice) -> np.ndarray:
        """
        Get a slice from the array along the first dimension.

        Parameters
        ----------
        s : FixedSlice
            The section of the array to get

        """
        if not hasattr(self, "_open") or not self._open:
            raise RuntimeError("Array access must be within context manager")
        else:
            return self._arrayslice(s.start, s.stop)

    def __enter__(self) -> None:
        """
        Enter the context.

        TODO
        """
        self._open = True


    def __exit__(self, *args) -> None:
        """
        Exit the context.

        TODO
        """
        self._open = False

    def _arrayslice(self, start: int, stop: int) -> np.ndarray:
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
