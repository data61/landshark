from collections import namedtuple

import tables

from landshark import image
from landshark.rowcache import RowCache


class ImageFeatures:

    def __init__(self, filename, cache_blocksize, cache_nblocks):
        self._hfile = tables.open_file(filename)
        x_coordinates = self._hfile.root.x_coordinates.read()
        y_coordinates = self._hfile.root.y_coordinates.read()
        height = self._hfile.root._v_attrs.height
        width = self._hfile.root._v_attrs.width
        spec = image.ImageSpec(width, height, x_coordinates, y_coordinates)
        self.image_spec = spec
        self.ord = Features(self._hfile.root.ordinal_data,
                            self._hfile.root.ordinal_data.attrs.missing_values,
                            cache_blocksize, cache_nblocks)
        self.cat = Features(self._hfile.root.categorical_data,
                            self._hfile.root.categorical_data.attrs.missing_values,
                            cache_blocksize, cache_nblocks)

    def pixel_indices(self, batchsize: int):
        pixel_it = image.coords_query(self.image_spec.width,
                                      self.image_spec.height, batchsize)
        return pixel_it


class Features:

    def __init__(self, carray, missing_values, cache_blocksize, cache_nblocks):
        self._carray = carray
        self._missing_values = missing_values
        self._cache = RowCache(carray, cache_blocksize, cache_nblocks)

    @property
    def nfeatures(self):
        return self._carray.atom.shape[0]

    @property
    def dtype(self):
        return self._carray.atom.dtype.base

    @property
    def missing_values(self):
        return self._missing_values

    def __call__(self, y, x_slice):
        return self._cache(y, x_slice)


class Targets:

    def __init__(self, filename, label):
        self._hfile = tables.open_file(filename)
        labels = self._hfile.root.targets.attrs.labels
        label_index = labels.index(label)
        # TODO don't read the whole file into memory
        self._data = self._hfile.root.targets[:, label_index]
        self.coordinates = self._hfile.root.coordinates

    def training(self, image_spec, batchsize):
        pixel_it = image.coords_training(self.coordinates,
                                         image_spec.x_coordinates,
                                         image_spec.y_coordinates, batchsize)
        start = 0
        for px, py in pixel_it:
            n = px.shape[0]
            stop = start + n
            s = slice(start, stop)
            target = self._data[s]
            start = stop
            yield px, py, target
