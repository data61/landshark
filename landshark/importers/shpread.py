"""Input/output routines for geo data types."""

from functools import partial
import itertools
import logging
import datetime

import numpy as np
import shapefile
# for mypy type checking
from typing import List, Tuple, Iterator, Union

from landshark import iteration
from landshark.importers.category import _Categories

log = logging.getLogger(__name__)


def _extract_type(python_type, field_length):
    if python_type is float:
        return np.float32
    elif python_type is int:
        return np.int32
    elif python_type is str:
        return "a" + str(field_length)
    elif python_type is datetime.date:
        return "a10"


def _get_record_info(shp):
    field_list = shp.fields[1:]
    labels, type_strings, nbytes, decimals = zip(*field_list)
    record0 = shp.record(0)
    types_from_data = [type(k) for k in record0]
    type_list = [_extract_type(t, l) for t, l in zip(types_from_data, nbytes)]
    return labels, type_list


def _get_dtype(labels, all_labels, all_dtypes):
    dtype_dict = dict(zip(all_labels, all_dtypes))
    dtype_set = {dtype_dict[l] for l in labels}
    if len(dtype_set) > 1:
        raise ValueError("Requested target labels have different types")
    dtype = dtype_set.pop()
    return dtype


def _to_array(record, indices, dtype):
    x_i = np.array([record[i] for i in indices], dtype=dtype)
    return x_i


def _get_indices(labels, all_labels):
    label_dict = dict(zip(all_labels, range(len(all_labels))))
    label_indices = [label_dict[k] for k in labels]
    return label_indices


def _to_coords(shape):
    result = shape.__geo_interface__["coordinates"]
    return result


def _convert_batch(b):
    coords, arrays = zip(*b)
    coords_x, coords_y = zip(*coords)
    coords_x = np.array(coords_x, dtype=np.float64)
    coords_y = np.array(coords_y, dtype=np.float64)
    data = np.vstack(arrays)
    return coords_x, coords_y, data

def _categorical_batch(b, categories):
    x, y, d = b
    dstar = categories.update(d)
    return x, y, dstar

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
    def __init__(self, filename: str, labels: List[str],
                 batchsize: int=100) -> None:
        """Construct an instance of ShapefileTargets."""
        self._sf = shapefile.Reader(filename)
        self.all_fields, self.all_dtypes = _get_record_info(self._sf)
        self.labels = labels
        self._label_indices = _get_indices(self.labels, self.all_fields)
        self.dtype = _get_dtype(self.labels, self.all_fields, self.all_dtypes)

        # if dtype is not float assume it's a classification task
        if self.dtype != np.float32 or self.dtype != np.float64:
            self.classification = True
            self._categories = _Categories([None] * len(self.labels))
        else:
            self.classification = False

        self._ntotal = self._sf.numRecords
        self.n = self._ntotal
        self.batchsize = batchsize
        self._seen_all_data = False

    def _data(self) -> Iterator[np.ndarray]:
        """Create an iterator for the shapefile data.

        This will return a batch of coordinates,  in order
        from the shapefile.

        Yields
        ------

        """
        for sr in self._sf.iterShapeRecords():
            coords = _to_coords(sr.shape)
            array = _to_array(sr.record, self._label_indices, self.dtype)
            yield coords, array
        self._seen_all_data = True

    def batches(self):
        list_batch_it = iteration.batch(self._data(), self.batchsize)
        batch_it = map(_convert_batch, list_batch_it)
        if self.classification:
            f = partial(_categorical_batch, categories=self._categories)
            res_batch_it = map(f, batch_it)
        else:
            res_batch_it = batch_it

        return res_batch_it

    @property
    def categorical_map(self):
        if not self._seen_all_data:
            raise RuntimeError("You must complete the data iterator before
                               having the category mappings")
        return self._categories.maps()
