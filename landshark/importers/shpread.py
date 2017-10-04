"""Input/output routines for geo data types."""

import logging

import numpy as np
import shapefile
# for mypy type checking
from typing import List, Tuple, Iterator

log = logging.getLogger(__name__)

# Typechecking aliases
ShpFieldsType = List[Tuple[str, str, int, int]]


def _shapefile_float_fields(fields: ShpFieldsType) \
        -> Iterator[Tuple[int, str]]:
    """Pull out the float fields from the shapefile field specification.

    Parameters
    ----------
    fields : ShpFieldsType
        The weird list-of-lists that shapefile uses to describe field types

    Returns
    -------
    result : Iterator[Tuple[int, str]]
        An iterator over (<index_number>, <name>) pairs of the float columns.

    """
    shapefields = [f[0] for f in fields[1:]]  # Skip DeletionFlag
    dtype_flags = [(f[1], f[2]) for f in fields[1:]]  # Skip DeletionFlag
    # http://www.dbase.com/Knowledgebase/INT/db7_file_fmt.htm
    # We're only going to support float types for now
    field_indices = [i for i, k in enumerate(dtype_flags) if k[0] == "N"]
    field_names = [shapefields[i] for i in field_indices]
    result = zip(field_indices, field_names)
    return result


class ShapefileTargets:
    """
    Targets for spatial inference backed by a shapefile.

    This class reads a shapefile with point information and provides
    iterators to the coordinates and the (float) records.

    Parameters
    ----------
    filename : str
        The shapefile (.shp)

    """

    def __init__(self, filename: str) -> None:
        """Construct an instance of ShapefileTargets."""
        self._sf = shapefile.Reader(filename)
        self.n = self._sf.numRecords
        self.dtype = np.float32
        float_fields = _shapefile_float_fields(self._sf.fields)
        self._field_indices, self.fields = zip(*float_fields)
        self._field_indices = list(self._field_indices)
        self.fields = list(self.fields)

    def coordinates(self) -> Iterator[np.ndarray]:
        """Create an iterator for the coordinate data.

        This will return a single coordinate from a point, in order
        from the shapefile.

        Returns
        -------
        res : Iterator[np.ndarray]
            An iterator over shape (2,) arrays of x,y coordinates.

        """
        for shape in self._sf.iterShapes():
            res = np.array(shape.__geo_interface__["coordinates"],
                           dtype=np.float64)
            yield res

    def ordinal_data(self) -> Iterator[np.ndarray]:
        """Create an iterator for the ordinal (target) data.

        This will return data at a single point per iteration,
        in order from the shapefile.

        Returns
        -------
        res : Iterator[np.ndarray]
            An iterator over shape (k,) of k records for each point.

        """
        for rec in self._sf.iterRecords():
            res = np.array([rec[i] for i in self._field_indices],
                           dtype=self.dtype)
            yield res
