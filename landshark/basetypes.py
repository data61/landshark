"""Baseclass for datatypes. Modron-light basically."""

import logging

import numpy as np
from typing import Union, Tuple, Optional, List, Sized

log = logging.getLogger(__name__)

# Definitions of numerical types used in the project.
OrdinalType = np.float32
CategoricalType = np.int32
NumericalType = Union[np.float32, np.int32]
MissingType = Optional[NumericalType]
CoordinateType = np.float64


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


# These value types are wrappers for ndarrays, which may have ordinal data,
# categorical data, or both.


class CategoricalValues:
    """
    Categorical (integer-valued) data array wrapper.

    Parameters
    ----------
    array : np.ndarray
        Numpy array of type CategoricalType.

    """

    def __init__(self, array: np.ndarray) -> None:
        """Initialise the object."""
        assert array.dtype == CategoricalType
        self.categorical = array


class OrdinalValues:
    """
    Ordinal (floating-point-valued) data array wrapper.

    Parameters
    ----------
    array : np.ndarray
        Numpy array of type OrdinalType.

    """

    def __init__(self, array: np.ndarray) -> None:
        """Initialise the object."""
        assert array.dtype == OrdinalType
        self.ordinal = array


class MixedValues(OrdinalValues, CategoricalValues):
    """
    Mixed (ordinal and categorical) data array wrapper.

    Parameters
    ----------
    ord_array : np.ndarray
        Numpy array of type OrdinalType.
    cat_array : np.ndarray
        Numpy array of type CategoricalType.

    """

    def __init__(self, ord_array: np.ndarray, cat_array: np.ndarray) -> None:
        """Initialise the object."""
        assert len(ord_array) == len(cat_array)
        OrdinalValues.__init__(self, ord_array)
        CategoricalValues.__init__(self, cat_array)


class CategoricalCoords(CategoricalValues):
    """
    Categorical values that have a point location in space.

    Parameters
    ----------
    array : np.ndarray
        The categorical values. Of shape (n, ...).
    coords : np.ndarray
        The coordinates for the categorical values., Of shape (n, ...).

    """

    def __init__(self, array: np.ndarray, coords: np.ndarray) -> None:
        """Initialise the object."""
        assert len(array) == len(coords)
        super().__init__(array)
        self.coords = coords


class OrdinalCoords(OrdinalValues):
    """
    Ordinal values that have a point location in space.

    Parameters
    ----------
    array : np.ndarray
        The ordinal values. Of shape (n, ...).
    coords : np.ndarray
        The coordinates for the ordinal values., Of shape (n, ...).

    """

    def __init__(self, array: np.ndarray, coords: np.ndarray) -> None:
        """Initialise the object."""
        assert len(array) == len(coords)
        super().__init__(array)
        self.coords = coords


class MixedCoords(MixedValues):
    """
    Mixed values that have a point location in space.

    Parameters
    ----------
    array : np.ndarray
        The ordinal values. Of shape (n, ...).
    coords : np.ndarray
        The coordinates for the mixed values., Of shape (n, ...).

    """

    def __init__(self, ord_array: np.ndarray, cat_array: np.ndarray,
                 coords: np.ndarray) -> None:
        """Initialise the object."""
        assert len(ord_array) == len(coords)
        assert len(cat_array) == len(coords)
        super().__init__(ord_array, cat_array)
        self.coords = coords


# Generic Value type (because doesn't have a common ancestor
Values = Union[CategoricalValues, OrdinalValues, MixedValues,
               CategoricalCoords, OrdinalCoords, MixedCoords]


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

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        """Perform the array slice. This gets overridden by children."""
        raise NotImplementedError


class OrdinalArraySource(ArraySource):
    """Array source for Ordinal data."""

    _dtype = OrdinalType


class CategoricalArraySource(ArraySource):
    """Array source for categorical data."""

    _dtype = CategoricalType


class CoordinateArraySource(ArraySource):
    """Array source for coordinate data."""

    _dtype = CoordinateType


class DataSource:
    """
    Abstract Data object that may have categorical, ordinal or mixed data.

    Has a notion of 'rows' that can be sliced, and Values contained in
    each row. Values can be ordinal categorical or mixed and with or without
    coordinates.

    """

    def __init__(self) -> None:
        """Initialise a variable to store the number of rows."""
        self._n = 0

    def slice(self, start: int, end: int) -> Values:
        """
        Return values from the data in a slice.

        Parameters
        ----------
        start : int
            A start index >=0 that is inclusive.
        end : int
            A stop index >= start that is exclusive.

        Returns
        -------
        values : Values
            An ordinal, categorical or mixed Values object containing the
            data in the requested slice.

        """
        values = self._slice(start, end)
        return values

    def _slice(self, start: int, end: int) -> Values:
        """Do the actual slicing. This is the fn that gets overridden."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of rows of the dataset."""
        return self._n


class OrdinalDataSource(DataSource):
    """
    Datasource for ordinal data.

    Parameters
    ----------
    source : OrdinalArraySource
        The array source providing the ordinal data.

    """

    def __init__(self, source: OrdinalArraySource) -> None:
        """Initialise the object."""
        self.ordinal = source
        self._n = self.ordinal.shape[0]

    def _slice(self, start: int, end: int) -> OrdinalValues:
        """Implement slice for the ordinal specialisation."""
        result = OrdinalValues(self.ordinal._arrayslice(start, end))
        return result


class CategoricalDataSource(DataSource):
    """
    Datasource for categorical data.

    Parameters
    ----------
    source: CategoricalArraySource
        The array source providing the categorical data.

    """

    def __init__(self, source: CategoricalArraySource) -> None:
        """Initialise the object."""
        self.categorical = source
        self._n = self.categorical.shape[0]

    def _slice(self, start: int, end: int) -> CategoricalValues:
        """Implement slice for the specialisation."""
        result = CategoricalValues(self.categorical._arrayslice(start, end))
        return result


class MixedDataSource(DataSource):
    """
    Datasource for mixed coordinate data.

    Parameters
    ----------
    ord_source: OrdinalArraySource
        The array source providing the ordinal data.
    cat_source: CategoricalArraySource
        The array source providing the categorical data.

    """

    def __init__(self, ordinal_source: OrdinalArraySource,
                 categorical_source: CategoricalArraySource) -> None:
        """Initialise the object."""
        self.ordinal = ordinal_source
        self.categorical = categorical_source
        assert ordinal_source.shape[0] == categorical_source.shape[0]
        self._n = ordinal_source.shape[0]

    def _slice(self, start: int, stop: int) -> MixedValues:
        """Implement slice for this specialisation."""
        result = MixedValues(self.ordinal._arrayslice(start, stop),
                             self.categorical._arrayslice(start, stop))
        return result


class OrdinalCoordSource(OrdinalDataSource):
    """
    Datasource for ordinal coordinate data.

    Parameters
    ----------
    source : OrdinalArraySource
        The array source providing the ordinal data.
    coord_source : CoordinateArraySource
        The array source providing the coordinate data. Must have the same
        length as source.

    """

    def __init__(self, ord_source: OrdinalArraySource,
                 coord_source: CoordinateArraySource) -> None:
        """Initialise the object."""
        super().__init__(ord_source)
        self.coordinates = coord_source

    def _slice(self, start: int, end: int) -> OrdinalCoords:
        """Implement slice for the specialisation."""
        result = OrdinalCoords(self.ordinal._arrayslice(start, end),
                               self.coordinates._arrayslice(start, end))
        return result


class CategoricalCoordSource(CategoricalDataSource):
    """
    Datasource for categorical coordinate data.

    Parameters
    ----------
    cat_source: CategoricalArraySource
        The array source providing the categorical data.
    coord_source : CoordinateArraySource
        The array source providing the coordinate data. Must have the same
        length as the cat_source.

    """

    def __init__(self, cat_source: CategoricalArraySource,
                 coord_source: CoordinateArraySource) -> None:
        """Initialise the object."""
        assert len(cat_source) == len(coord_source)
        super().__init__(cat_source)
        self.coordinates = coord_source

    def _slice(self, start: int, end: int) -> CategoricalCoords:
        """Implement slice for the specialisation."""
        result = CategoricalCoords(self.categorical._arrayslice(start, end),
                                   self.coordinates._arrayslice(start, end))
        return result


class MixedCoordSource(MixedDataSource):
    """
    Datasource for mixed coordinate data.

    Parameters
    ----------
    ord_source: OrdinalArraySource
        The array source providing the ordinal data.
    cat_source: CategoricalArraySource
        The array source providing the categorical data.
    coord_source : CoordinateArraySource
        The array source providing the coordinate data. Must have the same
        length as the cat_source.

    """

    def __init__(self, ord_source: OrdinalArraySource,
                 cat_source: CategoricalArraySource,
                 coord_source: CoordinateArraySource) -> None:
        """Initialise the object."""
        assert len(ord_source) == len(coord_source)
        assert len(cat_source) == len(coord_source)
        super().__init__(ord_source, cat_source)
        self.coordinates = coord_source

    def _slice(self, start: int, end: int) -> MixedCoords:
        """Implement slice for the specialisation."""
        result = MixedCoords(self.ordinal._arrayslice(start, end),
                             self.categorical._arrayslice(start, end),
                             self.coordinates._arrayslice(start, end))
        return result
