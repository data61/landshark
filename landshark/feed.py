"""Feeding iterators for training and querying data."""

from collections import namedtuple
from functools import partial

import numpy as np
from typing import Iterator, List, Tuple, Union, Any

from landshark import image
from landshark import patch
from landshark.patch import PatchRowRW, PatchMaskRowRW
from landshark.hread import ImageFeatures, Features

TrainingBatch = namedtuple("TrainingBatch", ["x_ord", "x_cat", "y"])
QueryBatch = namedtuple("QueryBatch", ["x_ord", "x_cat"])


def _training_fn(target_data, features: ImageFeatures, halfwidth: int) \
        -> TrainingBatch:
    """
    Get get training data from target data

    Parameters
    ----------
    features : ImageFeatures
        The "X" features (covariates) for training
    targets : Targets
        The "Y" targets for training
    batchsize : int
        The number of points to extract per call of the iterator
    halfwidth : int
        The half-width of image patches in X, ie number of additional
        pixels from centre

    Returns
    -------
    t : TrainingBatch

    """
    coords_x, coords_y, targets = target_data
    indices_x = image.world_to_image(coords_x,
                                     features.image_spec.x_coordinates)
    indices_y = image.world_to_image(coords_y,
                                     features.image_spec.y_coordinates)
    ord_marray, cat_marray = _read_batch(indices_x, indices_y,
                                         features, halfwidth)
    t = TrainingBatch(x_ord=ord_marray, x_cat=cat_marray, y=targets)
    return t


def training_data(target_it, features, halfwidth):
    training_it = map(lambda x: _training_fn(x, features, halfwidth),
                      target_it)
    return training_it


def query_data(indices: Iterator[Any], features: ImageFeatures,
               halfwidth: int) -> Iterator[QueryBatch]:
    """
    Create an iterator over batches of query data.

    Parameters
    ----------
    features : ImageFeatures
        The "X" features (covariates) for training
    batchsize : int
        The number of points to extract per call of the iterator
    halfwidth : int
        The half-width of image patches in X, ie number of additional
        pixels from centre

    Returns
    -------
    t : Iterator[TrainingBatch]
        An iterator that produces batches of X values for prediction

    """
    assert halfwidth >= 0

    def f(x):
        x_ord, x_cat = _read_batch(x[0], x[1], features, halfwidth)
        return QueryBatch(x_ord, x_cat)

    query_it = map(f, indices)
    return query_it


def _read(data: Features,
          patch_reads: List[PatchRowRW],
          mask_reads: List[PatchMaskRowRW],
          npatches: int,
          patchwidth: int,
          fill: Union[int, float, None]=0) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    init_f = np.empty if fill is None else partial(np.full, fill_value=fill)
    patch_data = init_f((npatches, data.nfeatures, patchwidth, patchwidth),
                        dtype=data.dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, :, r.yp, r.xp] = data(r.y, r.x).T

    for m in mask_reads:
        patch_mask[m.idx, :, m.yp, m.xp] = True

    for i, v in enumerate(data.missing_values):
        if v is not None:
            patch_mask[:, i, ...] |= (patch_data[:, i, ...] == v)

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray


def _read_batch(indices_x: np.ndarray, indices_y: np.ndarray,
                features: ImageFeatures, halfwidth: int) \
        -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Extract patches given a set of patch centre pixel indices."""
    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            halfwidth,
                                            features.image_spec.width,
                                            features.image_spec.height)
    npatches = indices_x.shape[0]
    patchwidth = 2 * halfwidth + 1

    ord_marray, cat_marray = None, None
    if features.ord is not None:
        ord_marray = _read(features.ord, patch_reads,
                           mask_reads, npatches, patchwidth)
    if features.cat is not None:
        cat_marray = _read(features.cat, patch_reads, mask_reads,
                           npatches, patchwidth)
    return ord_marray, cat_marray
