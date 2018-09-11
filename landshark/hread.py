"""Read features and targets from HDF5 files."""

from types import TracebackType
from typing import Tuple, Union

import numpy as np
import tables

from landshark.basetypes import (ArraySource, CategoricalArraySource,
                                 ContinuousArraySource)
from landshark.category import CategoryInfo
from landshark.image import ImageSpec


class H5ArraySource(ArraySource):

    _array_name = ""
    _label_name = ""

    def __init__(self, path: str) -> None:
        self._path = path
        with tables.open_file(self._path, "r") as hfile:
            carray = hfile.get_node("/" + self._array_name)
            self._shape = tuple(list(carray.shape) +
                                [carray.atom.dtype.shape[0]])
            self._missing = carray.attrs.missing
            array_cols = hfile.get_node("/" + self._label_name)
            self._columns = [s.decode() for s in array_cols.read()]
            self._native = carray.chunkshape[0]
            self._dtype = carray.atom.dtype.base

    def __enter__(self) -> None:
        self._hfile = tables.open_file(self._path, "r")
        self._carray = self._hfile.get_node("/" + self._array_name)
        if hasattr(self._hfile.root, "coordinates"):
            self._coords = self._hfile.root.coordinates
        super().__enter__()

    def __exit__(self, ex_type: type, ex_val: Exception,
                 ex_tb: TracebackType) -> None:
        self._hfile.close()
        del(self._carray)
        if hasattr(self, "_coords"):
            del(self._coords)
        del(self._hfile)
        super().__exit__(ex_type, ex_val, ex_tb)

    def _arrayslice(self, start: int, end: int) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        # TODO: Note this is bad because I'm changing the return type.

        data = self._carray[start:end]
        if hasattr(self, "_coords"):
            coords = self._coords[start:end]
            return data, coords
        else:
            return data


class ContinuousH5ArraySource(H5ArraySource, ContinuousArraySource):

    _array_name = "continuous_data"
    _label_name = "continuous_labels"


class CategoricalH5ArraySource(H5ArraySource, CategoricalArraySource):

    _array_name = "categorical_data"
    _label_name = "categorical_labels"


class H5Features:
    """Note unlike the array classes this isn't picklable."""

    def __init__(self, h5file: str) -> None:

        self.continuous, self.categorical, self.coordinates = None, None, None
        self._hfile = tables.open_file(h5file, "r")
        if hasattr(self._hfile.root, "continuous_data"):
            self.continuous = self._hfile.root.continuous_data
            self.continuous.mean = self._hfile.root.continuous_data.attrs.mean
            self.continuous.variance = \
                self._hfile.root.continuous_data.attrs.variance
            self.continuous.missing = self._hfile.root.continuous_data.attrs.missing
        if hasattr(self._hfile.root, "categorical_data"):
            self.categorical = self._hfile.root.categorical_data
            maps = self._hfile.root.categorical_mappings.read()
            counts = self._hfile.root.categorical_counts.read()
            self.categorical.missing = \
                self._hfile.root.categorical_data.attrs.missing
            self.categorical.maps = CategoryInfo(maps, counts)
        if self.continuous:
            self._n = len(self.continuous)
        if self.categorical:
            self._n = len(self.categorical)
        if self.continuous and self.categorical:
            assert len(self.continuous) == len(self.categorical)

    def __len__(self) -> int:
        return self._n

    # def rows(self, rows: FixedSlice) -> FeatureValues:
    #     con_data = None
    #     cat_data = None
    #     if self.continuous:
    #         con_data = self.continuous[rows.start:rows.stop]
    #     if self.categorical:
    #         cat_data = self.categorical[rows.start:rows.stop]
    #     return FeatureValues(con_data, cat_data)

    def __del__(self) -> None:
        self._hfile.close()


def read_image_spec(filename: str) -> ImageSpec:
    with tables.open_file(filename, mode="r") as h5file:
        x_coordinates = h5file.root.x_coordinates.read()
        y_coordinates = h5file.root.y_coordinates.read()
        crs = h5file.root._v_attrs.crs
    imspec = ImageSpec(x_coordinates, y_coordinates, crs)
    return imspec
