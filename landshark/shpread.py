"""Input/output routines for geo data types."""

import logging
import datetime

import numpy as np
import shapefile
# for mypy type checking
from typing import List

from landshark.basetypes import ArraySource, OrdinalArraySource, \
    CategoricalArraySource, OrdinalDataSource, CategoricalDataSource, \
    CoordinateArraySource

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


def _get_indices(labels, all_labels):
    label_dict = dict(zip(all_labels, range(len(all_labels))))
    label_indices = [label_dict[k] for k in labels]
    return label_indices


def _get_dtype(labels, all_labels, all_dtypes):
    dtype_dict = dict(zip(all_labels, all_dtypes))
    dtype_set = {dtype_dict[l] for l in labels}
    if len(dtype_set) > 1:
        raise ValueError("Requested target labels have different types")
    dtype = dtype_set.pop()
    return dtype


# TODO force this to be abstract. DONT USE!
class _ShpArraySource(ArraySource):
    def __init__(self, filename: str, labels: List[str],
                 random_seed: int) -> None:
        self._sf = shapefile.Reader(filename)
        all_fields, all_dtypes = _get_record_info(self._sf)
        self._columns = labels
        self._column_indices = _get_indices(self._columns, all_fields)
        self._shape = (self._sf.numRecords, len(labels))
        self._missing = [None] * self._shape[1]
        log.info("Shapefile contains {} records "
                 "of {} requested columns.".format(
                     self._shape[0], self._shape[1]))
        self._native = 1
        self._seed = 1
        rnd = np.random.RandomState(random_seed)
        self._perm = rnd.permutation(self._shape[0])

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        indices = self._perm[start: end]
        records = (self._sf.record(r) for r in indices)
        data = [[r[i] for i in self._column_indices] for r in records]
        array = np.array(data, dtype=self.dtype)
        return array


class OrdinalShpArraySource(_ShpArraySource, OrdinalArraySource):
    pass


class CategoricalShpArraySource(_ShpArraySource, CategoricalArraySource):
    pass


class CoordinateShpArraySource(CoordinateArraySource):

    def __init__(self, filename: str, random_seed: int)-> None:
        self._sf = shapefile.Reader(filename)
        self._shape = (self._sf.numRecords, 2)
        self._native = 1
        self._missing = [None, None]
        self._columns = ["X", "Y"]
        self._seed = 1
        rnd = np.random.RandomState(random_seed)
        self._perm = rnd.permutation(self._shape[0])

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        indices = self._perm[start: end]
        coords = [self._sf.shape(r).__geo_interface__["coordinates"]
                  for r in indices]
        array = np.array(coords, dtype=self.dtype)
        return array


class OrdinalShpSource(OrdinalDataSource):
    def __init__(self, filename: str, labels: List[str],
                 random_seed: int) -> None:
        source = OrdinalShpArraySource(filename, labels, random_seed)
        super().__init__(source)

class CategoricalShpSource(CategoricalDataSource):
    def __init__(self, filename: str, labels: List[str],
                 random_seed: int) -> None:
        source = CategoricalShpArraySource(filename, labels, random_seed)
        super().__init__(source)

class CoordinateShpSource(OrdinalDataSource):
    def __init__(self, filename: str, random_seed: int) -> None:
        source = CoordinateShpArraySource(filename, random_seed)
        super().__init__(source)


