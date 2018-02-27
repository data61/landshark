"""Read features and targets from HDF5 files."""

import numpy as np
import tables

from landshark.image import ImageSpec
from landshark.basetypes import ArraySource, OrdinalArraySource, \
    CategoricalArraySource, CoordinateArraySource, FeatureValues, FixedSlice


class H5ArraySource(ArraySource):

    _array_name = ""

    def __init__(self, path) -> None:
        self._path = path
        with tables.open_file(self._path, "r") as hfile:
            carray = hfile.get_node("/" + self._array_name)
            self._shape = tuple(list(carray.shape) +
                                [carray.atom.dtype.shape[0]])
            self._missing = carray.attrs.missing
            self._columns = carray.attrs.columns
            self._native = carray.chunkshape[0]
            self._dtype = carray.atom.dtype.base

    def __enter__(self):
        self._hfile = tables.open_file(self._path, "r")
        self._carray = self._hfile.get_node("/" + self._array_name)
        super().__enter__()

    def __exit__(self, *args):
        self._hfile.close()
        del(self._carray)
        del(self._hfile)
        super().__exit__()

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        data = self._carray[start:end]
        return data


class OrdinalH5ArraySource(H5ArraySource, OrdinalArraySource):
    _array_name = "ordinal_data"


class CategoricalH5ArraySource(H5ArraySource, CategoricalArraySource):
    _array_name = "categorical_data"


class CoordinateH5ArraySource(H5ArraySource, CoordinateArraySource):
    pass


class H5Features:
    """
    Note unlike the array classes this isn't picklable.
    """
    def __init__(self, h5file, normalise):

        self.normalised = normalise
        self.ordinal, self.categorical, self.coordinates = None, None, None
        self._hfile = tables.open_file(h5file, "r")
        if hasattr(self._hfile.root, "ordinal_data"):
            self.ordinal = self._hfile.root.ordinal_data
            self.ordinal.mean = self._hfile.root.ordinal_data.attrs.mean
            self.ordinal.variance = \
                self._hfile.root.ordinal_data.attrs.variance
        if hasattr(self._hfile.root, "categorical_data"):
            self.categorical = self._hfile.root.categorical_data
            maps = self._hfile.root.categorical_mappings.read()
            counts = self._hfile.root.categorical_counts.read()
            missing = self._hfile.root.categorical_data.attrs.missing
            self.categorical.maps = CategoryInfo(maps, counts, missing)
        if hasattr(self._hfile.root, "coordinates"):
            self.coordinates = self._hfile.root.coordinates
        if self.ordinal:
            self._n = len(self.ordinal)
        if self.categorical:
            self._n = len(self.categorical)
        if self.ordinal and self.categorical:
            assert len(self.ordinal) == len(self.categorical)
        if self.ordinal and self.coordinates:
            assert len(self.ordinal) == len(self.coordinates)
        if self.categorical and self.coordinates:
            assert len(self.categorical) == len(self.coordinates)

    def __len__(self):
        return self._n

    def __call__(self, s: FixedSlice):
        ord_data = None
        cat_data = None
        coord_data = None
        if self.ordinal:
            ord_data = self.ordinal(s)
        if self.categorical:
            cat_data = self.categorical(s)
        if self.coordinates:
            coord_data = self.coordinates(s)
        return FeatureValues(ord_data, cat_data, coord_data)

    def __del__(self):
        self._hfile.close()

def read_image_spec(filename):
    with tables.open_file(filename, mode="r") as h5file:
        x_coordinates = h5file.root.x_coordinates.read()
        y_coordinates = h5file.root.y_coordinates.read()
        crs = h5file.root._v_attrs.crs
    imspec = ImageSpec(x_coordinates, y_coordinates, crs)
    return imspec
