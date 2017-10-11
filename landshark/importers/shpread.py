"""Input/output routines for geo data types."""

import logging

import numpy as np
import shapefile
# for mypy type checking
from typing import List, Tuple, Iterator, Union

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
    indices : ndarray
        array of indices of shape coordinates to read

    """

    def __init__(self, filename: str, indices: np.array=None) -> None:
        """Construct an instance of ShapefileTargets."""
        self._sf = shapefile.Reader(filename)
        self.dtype = np.float32
        float_fields = _shapefile_float_fields(self._sf.fields)
        self._field_indices, self.fields = zip(*float_fields)
        self._field_indices = list(self._field_indices)
        self.fields = list(self.fields)
        self.indices = indices
        self.n = self._sf.numRecords if indices is None else len(indices)

    def coordinates(self) -> Iterator[np.ndarray]:
        """Create an iterator for the coordinate data.

        This will return a single coordinate from a point, in order
        from the shapefile.

        Yields
        ------
        res : Iterator[np.ndarray]
            An iterator over shape (2,) arrays of x,y coordinates.

        """
        if self.indices is None:
            shapes = self._sf.iterShapes()
        else:
            shapes = (self._sf.shape(i) for i in self.indices)

        for shape in shapes:
            res = np.array(shape.__geo_interface__["coordinates"],
                           dtype=np.float64)
            yield res

    def ordinal_data(self) -> Iterator[np.ndarray]:
        """Create an iterator for the ordinal (target) data.

        This will return data at a single point per iteration,
        in order from the shapefile.

        Yields
        ------
        res : Iterator[np.ndarray]
            An iterator over shape (k,) of k records for each point.

        """
        if self.indices is None:
            records = self._sf.iterRecords()
        else:
            records = (self._sf.record(i) for i in self.indices)

        for rec in records:
            res = np.array([rec[f] for f in self._field_indices],
                           dtype=self.dtype)
            yield res


def test_train_targets(
        filename: str,
        test_proportion: float,
        random_state: Union[int, None]=None
        ) -> Tuple[ShapefileTargets, ShapefileTargets]:
    """Create training and testing data shapefile readers.

    Parameters
    ----------
    filename : str
        The shapefile (.shp)
    test_proportion : float
        Fraction of data to make testing
    random_state : int, None
        Random state to use for splitting the data, None for no seed.

    Returns
    -------
    tr_read : ShapefileTargets
        the training data shape file reader
    ts_read : ShapefileTargets
        the testing data shape file reader
    """
    assert test_proportion > 0. and test_proportion < 1.0

    # Get length of data
    n = shapefile.Reader(filename).numRecords

    # Randomly choose training data
    rnd = np.random.RandomState(random_state)
    ts_ind = np.sort(rnd.choice(n, round(test_proportion * n), replace=False))

    # Create training indices
    tr_mask = np.ones(n, dtype=bool)
    tr_mask[ts_ind] = False
    tr_ind = np.where(tr_mask)[0]

    # Make readers
    tr_read = ShapefileTargets(filename, tr_ind)
    ts_read = ShapefileTargets(filename, ts_ind)

    return tr_read, ts_read
