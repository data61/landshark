"""write training data"""
from functools import partial
from itertools import groupby, count
import logging

import numpy as np
from typing import List, Union, Tuple
import tables

from landshark import patch
from landshark.multiproc import task_list
from landshark.basetypes import FixedSlice
from landshark.patch import PatchRowRW, PatchMaskRowRW
from landshark.iteration import batch_slices
from landshark import image
from landshark import tfwrite
from landshark.hread import H5Features, CategoricalH5ArraySource, \
    OrdinalH5ArraySource
from landshark.image import indices_strip
from landshark.serialise import serialise

log = logging.getLogger(__name__)

def _direct_read(array,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    nfeatures = array.atom.shape[0]
    dtype = array.atom.dtype.base
    patch_data = np.zeros((npatches, nfeatures, patchwidth, patchwidth),
                          dtype=dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, :, r.yp, r.xp] = array[r.y, r.x].T

    for m in mask_reads:
        patch_mask[m.idx, :, m.yp, m.xp] = True

    if array.missing is not None:
        patch_mask |= patch_data == array.missing

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray

def _cached_read(row_dict,
                 array,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int,
                 fill: Union[int, float, None]=0) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    nfeatures = array.atom.shape[0]
    dtype = array.atom.dtype.base
    patch_data = np.zeros((npatches, nfeatures, patchwidth, patchwidth),
                          dtype=dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, :, r.yp, r.xp] = row_dict[r.y][r.x].T

    for m in mask_reads:
        patch_mask[m.idx, :, m.yp, m.xp] = True

    if array.missing is not None:
        patch_mask |= patch_data == array.missing

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray

def _as_range(iterable):
    lst = list(iterable)
    if len(lst) > 1:
        return FixedSlice(start=lst[0], stop=(lst[-1] + 1))
    else:
        return FixedSlice(start=lst[0], stop=(lst[0] + 1))

def _slices_from_patches(patch_reads):
    rowlist = sorted(list(set((k.y for k in patch_reads))))
    slices = [_as_range(g) for _, g in
              groupby(rowlist, key=lambda n, c=count(): n - next(c))]
    return slices

def _get_rows(slices, patch_reads, array):
    # TODO make faster
    data_slices = [array[s.start:s.stop] for s in slices]
    data = {}
    for s, d in zip(slices, data_slices):
        for i, d_io in zip(range(s[0], s[1]), d):
            data[i] = d_io
    return data


class TrainingDataProcessor:

    def __init__(self, image_spec, feature_path, halfwidth):
        self.feature_path = feature_path
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source = None

    def __call__(self, values):
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        targets, coords = values
        coords_x, coords_y = coords.T
        indices_x = image.world_to_image(coords_x,
                                         self.image_spec.x_coordinates)
        indices_y = image.world_to_image(coords_y,
                                         self.image_spec.y_coordinates)
        patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                                self.halfwidth,
                                                self.image_spec.width,
                                                self.image_spec.height)
        npatches = indices_x.shape[0]
        patchwidth = 2 * self.halfwidth + 1
        ord_marray, cat_marray = None, None
        if self.feature_source.ordinal:
            ord_marray = _direct_read(self.feature_source.ordinal,
                                      patch_reads, mask_reads,
                                      npatches, patchwidth)
        if self.feature_source.categorical:
            cat_marray = _direct_read(self.feature_source.categorical,
                                      patch_reads, mask_reads,
                                      npatches, patchwidth)

        strings = serialise(ord_marray, cat_marray, targets)
        return strings


class QueryDataProcessor:

    def __init__(self, image_spec, feature_path, halfwidth):
        self.feature_path = feature_path
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source = None

    def __call__(self, indices):
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        indices_x, indices_y = indices
        patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                                self.halfwidth,
                                                self.image_spec.width,
                                                self.image_spec.height)
        patch_data_slices = _slices_from_patches(patch_reads)
        npatches = indices_x.shape[0]
        patchwidth = 2 * self.halfwidth + 1
        ord_marray, cat_marray = None, None
        if self.feature_source.ordinal:
            ord_data_cache = _get_rows(patch_data_slices, patch_reads,
                                       self.feature_source.ordinal)
            ord_marray = _cached_read(ord_data_cache,
                                      self.feature_source.ordinal,
                                      patch_reads, mask_reads, npatches,
                                      patchwidth)
        if self.feature_source.categorical:
            cat_data_cache = _get_rows(patch_data_slices, patch_reads,
                                       self.feature_source.categorical)
            cat_marray = _cached_read(cat_data_cache,
                                      self.feature_source.categorical,
                                      patch_reads, mask_reads, npatches,
                                      patchwidth)
        strings = serialise(ord_marray, cat_marray, None)
        return strings


def write_trainingdata(feature_path, target_path, image_spec, batchsize,
                       halfwidth, n_workers, output_directory, testfold, folds,
                       random_seed):

    log.info("Testing data is fold {} of {}".format(testfold, folds))
    log.info("Writing training data to tfrecord in {}-point batches".format(batchsize))
    with tables.open_file(target_path, "r") as tfile:
        categorical = hasattr(tfile.root, "categorical_data")
    target_src = CategoricalH5ArraySource(target_path) if categorical \
        else OrdinalH5ArraySource(target_path)
    n_rows = len(target_src)
    worker = TrainingDataProcessor(image_spec, feature_path, halfwidth)
    tasks = list(batch_slices(batchsize, n_rows))
    out_it = task_list(tasks, target_src, worker, n_workers)
    n_train = tfwrite.training(out_it, n_rows, output_directory, testfold,
                               folds, random_seed)
    return n_train


class _DummyReader:
    def __call__(self, x):
        return x

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def write_querydata(feature_path, image_spec, strip, total_strips, batchsize,
                    halfwidth, n_workers, output_directory, tag):
    true_batchsize = batchsize * image_spec.width
    log.info("Writing query data to tfrecord in {}-row batches".format(
        true_batchsize))
    reader_src = _DummyReader()
    it, n_total = indices_strip(image_spec, strip, total_strips,
                                true_batchsize)
    worker = QueryDataProcessor(image_spec, feature_path, halfwidth)
    tasks = list(it)
    out_it = task_list(tasks, reader_src, worker, n_workers)
    tfwrite.query(out_it, n_total, output_directory, tag)
