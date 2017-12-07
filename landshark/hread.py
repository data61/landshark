"""Read features and targets from HDF5 files."""

import numpy as np
import tables
from landshark.image import ImageSpec

from landshark.basetypes import ArraySource, OrdinalArraySource, \
    CategoricalArraySource, OrdinalDataSource, CategoricalDataSource, \
    MixedDataSource, OrdinalCoordSource, CategoricalCoordSource


class _H5ArraySource(ArraySource):

    def __init__(self, carray) -> None:
        self._shape = tuple(list(carray.shape) + [carray.atom.shape[0]])
        self._carray = carray
        self._missing = carray.attrs.missing
        self._columns = carray.attrs.columns
        self._native = carray.chunkshape[0]
        self._dtype = self._carray.atom.dtype.base

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        return self._carray[start:end]


class _OrdinalH5ArraySource(_H5ArraySource, OrdinalArraySource):
    pass


class _CategoricalH5ArraySource(_H5ArraySource, CategoricalArraySource):
    pass


class OrdinalH5Source(OrdinalDataSource):
    def __init__(self, h5file):
        self._hfile = tables.open_file(h5file, "r")
        source = _OrdinalH5ArraySource(self._hfile.root.ordinal_data)
        super().__init__(source)

class CategoricalH5Source(CategoricalDataSource):
    def __init__(self, h5file):
        self._hfile = tables.open_file(h5file, "r")
        source = _CategoricalH5ArraySource(self._hfile.root.categorical_data)
        super().__init__(source)

class MixedH5Source(MixedDataSource):
    def __init__(self, h5file):
        self._hfile = tables.open_file(h5file, "r")
        ord_source = _OrdinalH5ArraySource(self._hfile.root.ordinal_data)
        cat_source = _CategoricalH5ArraySource(self._hfile.root.categorical_data)
        super().__init__(ord_source, cat_source)

class OrdinalCoordH5Source(OrdinalCoordSource):
    def __init__(self, h5file):
        self._hfile = tables.open_file(h5file, "r")
        ord_source = _OrdinalH5ArraySource(self._hfile.root.ordinal_data)
        coord_source = _OrdinalH5ArraySource(self._hfile.root.coordinates)
        super().__init__(ord_source, coord_source)


class CategoricalCoordH5Source(CategoricalCoordSource):
    def __init__(self, h5file):
        self._hfile = tables.open_file(h5file, "r")
        cat_source = _CategoricalH5ArraySource(self._hfile.root.categorical_data)
        coord_source = _OrdinalH5ArraySource(self._hfile.root.coordinates)
        super().__init__(cat_source, coord_source)


def datatype(hfile):
    with tables.open_file(hfile, mode="r") as h5file:
        categorical = hasattr(h5file.root, "categorical_data")
        ordinal = hasattr(h5file.root, "ordinal_data")
        coords = hasattr(h5file.root, "coordinates")

    if coords and ordinal:
        return OrdinalCoordH5Source
    elif coords and categorical:
        return CategoricalCoordH5Source
    elif ordinal and categorical:
        return MixedH5Source
    elif ordinal:
        return OrdinalH5Source
    elif categorical:
        return CategoricalH5Source
    else:
        return ValueError("H5 file missing tables for datatype conversion.")


def read_image_spec(filename):
    with tables.open_file(filename, mode="r") as h5file:
        x_coordinates = h5file.root.x_coordinates.read()
        y_coordinates = h5file.root.y_coordinates.read()
        crs = h5file.root._v_attrs.crs
    imspec = ImageSpec(x_coordinates, y_coordinates, crs)
    return imspec
