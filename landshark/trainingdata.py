"""write training data"""
from functools import partial
from itertools import groupby, count

import numpy as np
from typing import List, Union, Tuple

from landshark import patch
from landshark.multiproc import task_list
from landshark.basetypes import FixedSlice
from landshark.patch import PatchRowRW, PatchMaskRowRW
from landshark.iteration import batch_slices
from landshark import image
from landshark import tfwrite
from landshark.hread import H5Features
from landshark.image import indices_strip
from landshark.serialise import serialise


def _read(row_dict,
          source,
          patch_reads: List[PatchRowRW],
          mask_reads: List[PatchMaskRowRW],
          npatches: int,
          patchwidth: int,
          fill: Union[int, float, None]=0) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    nfeatures = source.shape[-1]
    dtype = source.dtype
    init_f = np.empty if fill is None else partial(np.full, fill_value=fill)
    patch_data = init_f((npatches, nfeatures, patchwidth, patchwidth),
                        dtype=dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, :, r.yp, r.xp] = row_dict[r.y][r.x].T

    for m in mask_reads:
        patch_mask[m.idx, :, m.yp, m.xp] = True

    for i, v in enumerate(source.missing):
        if v is not None:
            patch_mask[:, i, ...] |= (patch_data[:, i, ...] == v)

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray


def _as_range(iterable):
    lst = list(iterable)
    if len(lst) > 1:
        return FixedSlice(start=lst[0], stop=(lst[-1] + 1))
    else:
        return FixedSlice(start=lst[0], stop=(lst[0] + 1))

def _get_rows(patch_reads, source):
    # TODO make faster
    rowlist = sorted(list(set((k.y for k in patch_reads))))
    slices = [_as_range(g) for _, g in
              groupby(rowlist, key=lambda n, c=count(): n - next(c))]
    data_slices = [source(s) for s in slices]
    ord_data = {}
    cat_data = {}
    if source.categorical is None:
        for s, d in zip(slices, data_slices):
            for i, d_io in zip(range(s[0], s[1]), d.ordinal):
                ord_data[i] = d_io
    elif source.ordinal is None:
        for s, d in zip(slices, data_slices):
            for i, d_ic in zip(range(s[0], s[1]), d.categorical):
                cat_data[i] = d_ic
    else:
        for s, d in zip(slices, data_slices):
            for i, d_io, d_ic in zip(range(s[0], s[1]),
                                     d.ordinal, d.categorical):
                ord_data[i] = d_io
                cat_data[i] = d_ic

    if len(ord_data) == 0:
        ord_data = None
    if len(cat_data) == 0:
        cat_data = None
    return ord_data, cat_data


class TrainingDataProcessor:

    def __init__(self, image_spec, feature_path, halfwidth,
                 normalise_features):
        self.feature_path = feature_path
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source = None
        self.normalise = normalise_features

    def __call__(self, values):
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path, self.normalise)
        coords_x, coords_y = values.coordinates.T
        targets = values.ordinal if values.ordinal is not None \
            else values.categorical

        indices_x = image.world_to_image(coords_x,
                                         self.image_spec.x_coordinates)
        indices_y = image.world_to_image(coords_y,
                                         self.image_spec.y_coordinates)
        patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                                self.halfwidth,
                                                self.image_spec.width,
                                                self.image_spec.height)
        ord_data, cat_data = _get_rows(patch_reads, self.feature_source)
        npatches = indices_x.shape[0]
        patchwidth = 2 * self.halfwidth + 1
        ord_marray, cat_marray = None, None
        if ord_data is not None:
            ord_marray = _read(ord_data, self.feature_source.ordinal,
                               patch_reads, mask_reads, npatches, patchwidth)
        if cat_data is not None:
            cat_marray = _read(cat_data, self.feature_source.categorical,
                               patch_reads, mask_reads, npatches, patchwidth)
        strings = serialise(ord_marray, cat_marray, targets)
        return strings


class QueryDataProcessor:

    def __init__(self, image_spec, feature_file, halfwidth):
        self.feature_file = feature_file
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source = None

    def __call__(self, indices):
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_file)

        indices_x, indices_y = indices
        patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                                self.halfwidth,
                                                self.image_spec.width,
                                                self.image_spec.height)
        ord_data, cat_data = _get_rows(patch_reads, self.feature_source)
        npatches = indices_x.shape[0]
        patchwidth = 2 * self.halfwidth + 1
        ord_marray, cat_marray = None, None
        if ord_data is not None:
            ord_marray = _read(ord_data, self.feature_source.ordinal,
                               patch_reads, mask_reads, npatches, patchwidth)
        if cat_data is not None:
            cat_marray = _read(cat_data, self.feature_source.categorical,
                               patch_reads, mask_reads, npatches, patchwidth)
        strings = serialise(ord_marray, cat_marray, None)
        return strings


def write_trainingdata(feature_path, targets, image_spec, batchsize,
                       halfwidth, n_workers, output_directory, testfold, folds,
                       random_seed, normalise_x, normalise_y):

    target_src = H5Features(targets, normalise_y)
    n_rows = len(target_src)
    worker = TrainingDataProcessor(image_spec, feature_path, halfwidth,
                                   normalise_x)
    tasks = list(batch_slices(batchsize, n_rows))
    out_it = task_list(tasks, target_src, worker, n_workers)
    next(out_it)
    n_train = tfwrite.training(out_it, n_rows, output_directory, testfold,
                               folds, random_seed)
    return n_train


class _DummyReader:
    def __call__(self, x):
        return x


def write_querydata(features, image_spec, strip, total_strips, batchsize,
                    halfwidth, n_workers, output_directory, tag):

    it, n_total = indices_strip(image_spec, strip, total_strips, batchsize)

    reader_spec = ClassSpec(_DummyReader)
    worker_spec = ClassSpec(QueryDataProcessor,
                            [image_spec, features, halfwidth])
    tasks = list(it)
    out_it = task_list(tasks, reader_spec, worker_spec, n_workers)
    tfwrite.query(out_it, n_total, output_directory, tag)
