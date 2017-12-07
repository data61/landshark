"""write training data"""
from functools import partial
from itertools import groupby, count

import numpy as np
from typing import List, Union, Tuple

from landshark import patch
from landshark.patch import PatchRowRW, PatchMaskRowRW
from landshark.iteration import batch_slices
from landshark import image
from landshark import tfwrite
from landshark.hread import datatype
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
    l = list(iterable)
    if len(l) > 1:
        return (l[0], l[-1] + 1)
    else:
        return (l[0], l[0] + 1)

def _get_rows(patch_reads, source):
    # TODO make faster
    rowlist = sorted(list(set((k.y for k in patch_reads))))
    slices = [_as_range(g) for _, g in
              groupby(rowlist, key=lambda n, c=count(): n - next(c))]
    data_slices = [source.slice(start, end) for start, end in slices]
    ord_data = {}
    cat_data = {}
    for s, d in zip(slices, data_slices):
        if hasattr(d, "ordinal"):
            for i, d_i in zip(range(s[0], s[1]), d.ordinal):
                ord_data[i] = d_i
        if hasattr(d, "categorical"):
            for i, d_i in zip(range(s[0], s[1]), d.categorical):
                cat_data[i] = d_i
    if len(ord_data) == 0:
        ord_data = None
    if len(cat_data) == 0:
        cat_data = None
    return ord_data, cat_data


class TrainingDataProcessor:

    def __init__(self, image_spec, feature_class, feature_file, halfwidth):
        self.feature_class = feature_class
        self.feature_file = feature_file
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source = None

    def __call__(self, values):
        if not self.feature_source:
            self.feature_source = self.feature_class(self.feature_file)

        coords_x, coords_y = values.coords.T
        targets = values.ordinal if hasattr(values, "ordinal") \
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

    def __init__(self, image_spec, feature_class, feature_file, halfwidth):
        self.feature_class = feature_class
        self.feature_file = feature_file
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source = None

    def __call__(self, indices):
        if not self.feature_source:
            self.feature_source = self.feature_class(self.feature_file)

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


def write_trainingdata(features, targets, image_spec, batchsize,
                       halfwidth, pool, output_directory, test_frac,
                       random_seed):

    feature_class = datatype(features)
    target_source = datatype(targets)(targets)

    f = TrainingDataProcessor(image_spec, feature_class, features, halfwidth)
    n_rows = len(target_source)
    it = batch_slices(batchsize, n_rows)
    data_it = ((target_source.slice(start, end)) for start, end in it)
    out_it = pool.imap(f, data_it)
    n_total = len(target_source)
    n_train = tfwrite.training(out_it, n_total, output_directory, test_frac,
                               random_seed)
    return n_train


def write_querydata(features, image_spec, strip, total_strips, batchsize,
                    halfwidth, pool, output_directory, tag):

    feature_class = datatype(features)
    it, n_total = indices_strip(image_spec, strip, total_strips, batchsize)

    f = QueryDataProcessor(image_spec, feature_class, features, halfwidth)
    out_it = pool.imap(f, it)
    tfwrite.query(out_it, n_total, output_directory, tag)
