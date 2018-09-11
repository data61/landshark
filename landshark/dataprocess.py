"""Process training and query data."""

import logging
from itertools import count, groupby
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import tables

from landshark import patch
from landshark.basetypes import ArraySource, FixedSlice, Worker
from landshark.hread import H5Features
from landshark.image import ImageSpec, world_to_image
from landshark.kfold import KFolds
from landshark.patch import PatchMaskRowRW, PatchRowRW
from landshark.serialise import serialise

log = logging.getLogger(__name__)


def _direct_read(array: tables.CArray,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int,
                 active_cols: np.ndarray) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    assert active_cols.shape[0] == array.atom.shape[0]
    nfeatures = np.sum(active_cols)
    dtype = array.atom.dtype.base
    patch_data = np.zeros((npatches, nfeatures, patchwidth, patchwidth),
                          dtype=dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, :, r.yp, r.xp] = array[r.y, r.x][:, active_cols].T

    for m in mask_reads:
        patch_mask[m.idx, :, m.yp, m.xp] = True

    if array.missing is not None:
        patch_mask |= patch_data == array.missing

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray


def _cached_read(row_dict: Dict[int, np.ndarray],
                 array: tables.CArray,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int,
                 active_cols: np.ndarray) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    assert active_cols.shape[0] == array.atom.shape[0]
    nfeatures = np.sum(active_cols)
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


def _as_range(iterable: Iterator[int]) -> FixedSlice:
    lst = list(iterable)
    if len(lst) > 1:
        return FixedSlice(start=lst[0], stop=(lst[-1] + 1))
    else:
        return FixedSlice(start=lst[0], stop=(lst[0] + 1))


def _slices_from_patches(patch_reads: List[PatchRowRW]) -> List[FixedSlice]:
    rowlist = sorted(list({k.y for k in patch_reads}))

    def _get(n: int, c: Iterator[int]=count()) -> int:
        res = n - next(c)
        return res

    slices = [_as_range(g) for _, g in groupby(rowlist, key=_get)]
    return slices


def _get_rows(slices: List[FixedSlice], array: tables.CArray,
              active_cols: np.ndarray) -> Dict[int, np.ndarray]:
    # TODO make faster
    data_slices = [array[s.start:s.stop][..., active_cols] for s in slices]
    data = {}
    for s, d in zip(slices, data_slices):
        for i, d_io in zip(range(s[0], s[1]), d):
            data[i] = d_io
    return data


def _process_training(coords: np.ndarray,
                      feature_source: H5Features, image_spec: ImageSpec,
                      halfwidth: int, active_cons: np.ndarray,
                      active_cats: np.ndarray) -> \
        Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    coords_x, coords_y = coords.T
    indices_x = world_to_image(coords_x, image_spec.x_coordinates)
    indices_y = world_to_image(coords_y, image_spec.y_coordinates)
    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            halfwidth,
                                            image_spec.width,
                                            image_spec.height)
    npatches = indices_x.shape[0]
    patchwidth = 2 * halfwidth + 1
    con_marray, cat_marray = None, None
    if feature_source.continuous:
        con_marray = _direct_read(feature_source.continuous,
                                  patch_reads, mask_reads,
                                  npatches, patchwidth, active_cons)
    if feature_source.categorical:
        cat_marray = _direct_read(feature_source.categorical,
                                  patch_reads, mask_reads,
                                  npatches, patchwidth, active_cats)
    return con_marray, cat_marray


def _process_query(indices: Tuple[np.ndarray, np.ndarray],
                   feature_source: H5Features,
                   image_spec: ImageSpec, halfwidth: int,
                   active_con: np.ndarray, active_cat: np.ndarray) \
        -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    indices_x, indices_y = indices
    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            halfwidth,
                                            image_spec.width,
                                            image_spec.height)
    patch_data_slices = _slices_from_patches(patch_reads)
    npatches = indices_x.shape[0]
    patchwidth = 2 * halfwidth + 1
    con_marray, cat_marray = None, None
    if feature_source.continuous:
        con_data_cache = _get_rows(patch_data_slices,
                                   feature_source.continuous,
                                   active_con)
        con_marray = _cached_read(con_data_cache,
                                  feature_source.continuous,
                                  patch_reads, mask_reads, npatches,
                                  patchwidth, active_con)
    if feature_source.categorical:
        cat_data_cache = _get_rows(patch_data_slices,
                                   feature_source.categorical,
                                   active_cat)
        cat_marray = _cached_read(cat_data_cache,
                                  feature_source.categorical,
                                  patch_reads, mask_reads, npatches,
                                  patchwidth, active_cat)
    return con_marray, cat_marray


class SourceMetadata(NamedTuple):
    name: str
    feature_path: str
    target_src: ArraySource
    image_spec: ImageSpec
    halfwidth: int
    folds: KFolds
    active_cons: np.ndarray
    active_cats: np.ndarray


class TrainingDataProcessor(Worker):

    def __init__(self, tinfo: SourceMetadata) -> None:
        self.feature_path = tinfo.feature_path
        self.halfwidth = tinfo.halfwidth
        self.image_spec = tinfo.image_spec
        self.feature_source: Optional[H5Features] = None
        self.active_cons = tinfo.active_cons
        self.active_cats = tinfo.active_cats

    def __call__(self, values: Tuple[np.ndarray, np.ndarray]) -> \
            Tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ndarray]:
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        targets, coords = values
        con_marray, cat_marray = _process_training(coords, self.feature_source,
                                                   self.image_spec,
                                                   self.halfwidth,
                                                   self.active_cons,
                                                   self.active_cats)
        return con_marray, cat_marray, targets


class SerialisingTrainingDataProcessor(Worker):

    def __init__(self, tinfo: SourceMetadata) -> None:
        self.proc = TrainingDataProcessor(tinfo)

    def __call__(self, values: Tuple[np.ndarray, np.ndarray]) -> \
            List[bytes]:
        con_marray, cat_marray, targets = self.proc(values)
        strings = serialise(con_marray, cat_marray, targets)
        return strings


class QueryDataProcessor(Worker):

    def __init__(self, image_spec: ImageSpec, feature_path: str,
                 halfwidth: int, active_con: np.ndarray,
                 active_cat: np.ndarray) -> None:
        self.feature_path = feature_path
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source: Optional[H5Features] = None
        self.active_con = active_con
        self.active_cat = active_cat

    def __call__(self, indices: Tuple[np.ndarray, np.ndarray]) -> \
            Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        con_marray, cat_marray = _process_query(indices, self.feature_source,
                                                self.image_spec,
                                                self.halfwidth,
                                                self.active_con,
                                                self.active_cat)
        return con_marray, cat_marray


class SerialisingQueryDataProcessor(Worker):

    def __init__(self, image_spec: ImageSpec, feature_path: str,
                 halfwidth: int, active_con: np.ndarray,
                 active_cat: np.ndarray) -> None:
        self.proc = QueryDataProcessor(image_spec, feature_path, halfwidth,
                                       active_con, active_cat)

    def __call__(self, indices: Tuple[np.ndarray, np.ndarray]) -> \
            List[bytes]:
        con_marray, cat_marray = self.proc(indices)
        strings = serialise(con_marray, cat_marray, None)
        return strings
