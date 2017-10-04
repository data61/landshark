from collections import namedtuple

import numpy as np
from typing import Iterator

from landshark import patch
from landshark.hread import Features, Targets

TrainingBatch = namedtuple("TrainingBatch", ["x_ord", "x_cat", "y"])
QueryBatch = namedtuple("QueryBatch", ["x_ord", "x_cat"])


def training_data(features: Features, targets: Targets, batchsize, halfwidth) \
        -> Iterator[TrainingBatch]:

    it = targets.training(features.image_spec, batchsize)
    for x_indices, y_indices, target_batch in it:
        ord_marray, cat_marray = _read_batch(x_indices, y_indices,
                                             features, halfwidth)
        t = TrainingBatch(x_ord=ord_marray, x_cat=cat_marray, y=target_batch)
        yield t

def query_data(features: Features, batchsize, halfwidth):

    it = features.pixel_indices(batchsize)
    for x_indices, y_indices in it:
        ord_marray, cat_marray = _read_batch(x_indices, y_indices,
                                             features, halfwidth)
        b = QueryBatch(x_ord=ord_marray, x_cat=cat_marray)
        yield b


def _read(data, patch_reads, mask_reads, n, patchwidth):
    patch_data = np.empty((n, patchwidth, patchwidth, data.nfeatures),
                          dtype=data.dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, r.yp, r.xp] = data(r.y, r.x)

    for r in mask_reads:
        patch_mask[r.idx, r.yp, r.xp] = True

    for i, v in enumerate(data.missing_values):
        if v is not None:
            patch_mask[..., i] |= (patch_data[..., i] == v)

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray


def _read_batch(indices_x, indices_y, features, halfwidth):

    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            halfwidth,
                                            features.image_spec.width,
                                            features.image_spec.height)
    n = indices_x.shape[0]
    patchwidth = 2 * halfwidth + 1

    ord_marray = _read(features.ord, patch_reads, mask_reads, n, patchwidth)
    cat_marray = _read(features.cat, patch_reads, mask_reads, n, patchwidth)
    return ord_marray, cat_marray

