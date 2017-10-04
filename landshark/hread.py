# from image import coords_query, coords_training
# from feed import read_batch
from collections import namedtuple

import numpy as np
import tables
from typing import Iterator

from landshark import image
from landshark.rowcache import RowCache
from landshark.feed import read_batch


# def read_targets(xfile, yfile, target_label, batchsize):
#     x_pixel_array = xfile.root.x_coordinates.read()
#     y_pixel_array = xfile.root.y_coordinates.read()
#     coords_it = coords_training(yfile.root.coordinates, x_pixel_array,
#                                 y_pixel_array, batchsize)
#     labels = yfile.root.targets.attrs.labels
#     targets = yfile.root.targets.read()
#     Y = targets[:, labels.index(target_label)]
#     return coords_it, Y


# def read_features(xfile, ord_cache, cat_cache, batchsize, coords_it=None):
#     image_height = xfile.root._v_attrs.height
#     image_width = xfile.root._v_attrs.width
#     if coords_it is None:
#         coords_it = coords_query(image_width, image_height, batchsize)
#     data_batches = (read_batch(cx, cy, xfile, ord_cache, cat_cache)
#                     for cx, cy in coords_it)
#     return data_batches


class ImageSpec:
    def __init__(self, width: int, height: int, x_coordinates: np.ndarray,
                 y_coordinates: np.ndarray):
        self.width = width
        self.height = height
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates


class ImageFeatures:

    def __init__(self, filename, cache_blocksize, cache_nblocks):
        self._hfile = tables.open_file(filename)
        x_coordinates = self._hfile.root.x_coordinates.read()
        y_coordinates = self._hfile.root.y_coordinates.read()
        height = self._hfile.root._v_attrs.height
        width = self._hfile.root._v_attrs.width
        spec = ImageSpec(width, height, x_coordinates, y_coordinates)
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


TrainingBatch = namedtuple("TrainingBatch", ["x_ord", "x_cat", "y"])
QueryBatch = namedtuple("QueryBatch", ["x_ord", "x_cat"])

def training_data(features: Features, targets: Targets, batchsize, halfwidth) \
        -> Iterator[TrainingBatch]:

    it = targets.training(features.image_spec, batchsize)
    for x_indices, y_indices, target_batch in it:
        ord_marray, cat_marray = read_batch(x_indices, y_indices,
                                            features, halfwidth)
        t = TrainingBatch(x_ord=ord_marray, x_cat=cat_marray, y=target_batch)
        yield t

def query_data(features: Features, batchsize, halfwidth):

    it = features.pixel_indices(batchsize)
    for x_indices, y_indices in it:
        ord_marray, cat_marray = read_batch(x_indices, y_indices,
                                            features, halfwidth)
        b = QueryBatch(x_ord=ord_marray, x_cat=cat_marray)
        yield b
