"""Feeding iterators for training and querying data."""

from itertools import chain
from collections import namedtuple

import numpy as np
import tensorflow as tf
from typing import Iterator, List, Tuple

from landshark import patch
from landshark.patch import PatchRowRW, PatchMaskRowRW
from landshark.hread import ImageFeatures, Features, Targets

TrainingBatch = namedtuple("TrainingBatch", ["x_ord", "x_cat", "y"])
QueryBatch = namedtuple("QueryBatch", ["x_ord", "x_cat"])


class SliceTrainingData:

    def __init__(self, data: Iterator[TrainingBatch]) -> None:
        peek_d = next(data)

        self.types = (
            tf.as_dtype(peek_d.x_ord.dtype),
            tf.as_dtype(peek_d.x_cat.dtype),
            tf.as_dtype(peek_d.y.dtype)
            )

        self.shapes = (
            peek_d.x_ord.shape[1:],
            peek_d.x_cat.shape[1:],
            peek_d.y.shape[1:]
            )

        self.data = chain([peek_d], data)

    def __call__(self) -> Iterator[TrainingBatch]:
        # TODO deal with the masks!
        for d in self.data:
            for xo, xc, y in zip(d.x_ord, d.x_cat, d.y):
                tslice = TrainingBatch(xo, xc, y)
                yield tslice


def training_data(features: ImageFeatures, targets: Targets, batchsize: int,
                  halfwidth: int, epochs: int=1, flatten: bool=False) \
        -> Iterator[TrainingBatch]:
    """
    Create an iterator over batches of training data.

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
    epochs : int
        Number of times to repeat yielding the training dataset
    flatten : bool
        Flatten the training data features from patches into arrays of shape
        (batchsize, D)

    Yields
    ------
    t : Iterator[TrainingBatch]
        An iterator that produces batches of x,y pairs

    """
    assert batchsize > 0
    assert halfwidth >= 0

    for _ in range(epochs):
        it = targets.training(features.image_spec, batchsize)
        for x_indices, y_indices, target_batch in it:
            ord_marray, cat_marray = _read_batch(x_indices, y_indices,
                                                 features, halfwidth)
            if flatten:
                ord_marray = np.ma.reshape(ord_marray, [len(ord_marray), -1])
                cat_marray = np.ma.reshape(cat_marray, [len(cat_marray), -1])

            t = TrainingBatch(x_ord=ord_marray, x_cat=cat_marray,
                              y=target_batch)
            yield t


def query_data(features: ImageFeatures, batchsize: int, halfwidth: int) \
        -> Iterator[QueryBatch]:
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
    assert batchsize > 0
    assert halfwidth >= 0

    it = features.pixel_indices(batchsize)
    for x_indices, y_indices in it:
        ord_marray, cat_marray = _read_batch(x_indices, y_indices,
                                             features, halfwidth)
        b = QueryBatch(x_ord=ord_marray, x_cat=cat_marray)
        yield b


def _read(data: Features,
          patch_reads: List[PatchRowRW],
          mask_reads: List[PatchMaskRowRW],
          npatches: int,
          patchwidth: int,
          fill: bool=False) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    init_f = np.zeros if fill else np.empty
    patch_data = init_f((npatches, patchwidth, patchwidth, data.nfeatures),
                        dtype=data.dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, r.yp, r.xp] = data(r.y, r.x)

    for m in mask_reads:
        patch_mask[m.idx, m.yp, m.xp] = True

    for i, v in enumerate(data.missing_values):
        if v is not None:
            patch_mask[..., i] |= (patch_data[..., i] == v)

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

    ord_marray = _read(features.ord, patch_reads,
                       mask_reads, npatches, patchwidth)
    cat_marray = _read(features.cat, patch_reads, mask_reads,
                       npatches, patchwidth)
    return ord_marray, cat_marray
