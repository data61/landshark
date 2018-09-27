"""Process training and query data."""

import logging
from itertools import count, groupby
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import tables

from landshark import patch
from landshark.basetypes import ArraySource, FixedSlice, Worker
from landshark.hread import H5Features
from landshark.image import ImageSpec, world_to_image, image_to_world
from landshark.kfold import KFolds
from landshark.patch import PatchMaskRowRW, PatchRowRW
from landshark.serialise import serialise, DataArrays

log = logging.getLogger(__name__)


class SourceMetadata(NamedTuple):
    name: str
    feature_path: str
    target_src: Optional[ArraySource]
    image_spec: ImageSpec
    halfwidth: int
    folds: KFolds


def _direct_read(array: tables.CArray,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    nfeatures = array.atom.shape[0]
    dtype = array.atom.dtype.base
    patch_data = np.zeros((npatches, patchwidth, patchwidth, nfeatures),
                          dtype=dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, r.yp, r.xp] = array[r.y, r.x]

    for m in mask_reads:
        patch_mask[m.idx, m.yp, m.xp] = True

    if array.missing is not None:
        patch_mask |= patch_data == array.missing

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray


def _cached_read(row_dict: Dict[int, np.ndarray],
                 array: tables.CArray,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int) -> np.ma.MaskedArray:
    """Build patches from a data source given the read/write operations."""
    assert npatches > 0
    assert patchwidth > 0
    nfeatures = array.atom.shape[0]
    dtype = array.atom.dtype.base
    patch_data = np.zeros((npatches, patchwidth, patchwidth, nfeatures),
                          dtype=dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, r.yp, r.xp] = row_dict[r.y][r.x]

    for m in mask_reads:
        patch_mask[m.idx, m.yp, m.xp] = True

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


def _get_rows(slices: List[FixedSlice], array: tables.CArray) \
        -> Dict[int, np.ndarray]:
    # TODO make faster
    data_slices = [array[s.start:s.stop] for s in slices]
    data = {}
    for s, d in zip(slices, data_slices):
        for i, d_io in zip(range(s[0], s[1]), d):
            data[i] = d_io
    return data


def _process_training(coords: np.ndarray,
                      targets: np.ndarray,
                      feature_source: H5Features,
                      rec: SourceMetadata) -> DataArrays:
    coords_x, coords_y = coords.T
    indices_x = world_to_image(coords_x, rec.image_spec.x_coordinates)
    indices_y = world_to_image(coords_y, rec.image_spec.y_coordinates)
    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            rec.halfwidth,
                                            rec.image_spec.width,
                                            rec.image_spec.height)
    npatches = indices_x.shape[0]
    patchwidth = 2 * rec.halfwidth + 1
    con_marray, cat_marray = None, None
    if feature_source.continuous:
        con_marray = _direct_read(feature_source.continuous,
                                  patch_reads, mask_reads,
                                  npatches, patchwidth)
    if feature_source.categorical:
        cat_marray = _direct_read(feature_source.categorical,
                                  patch_reads, mask_reads,
                                  npatches, patchwidth)
    indices = np.vstack((indices_x, indices_y)).T
    output = DataArrays(con_marray, cat_marray, targets, coords, indices)
    return output


def _process_query(indices: np.ndarray,
                   feature_source: H5Features,
                   rec: SourceMetadata) -> DataArrays:
    indices_x, indices_y = indices.T
    coords_x = image_to_world(indices_x, rec.image_spec.x_coordinates)
    coords_y = image_to_world(indices_y, rec.image_spec.y_coordinates)
    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            rec.halfwidth,
                                            rec.image_spec.width,
                                            rec.image_spec.height)
    patch_data_slices = _slices_from_patches(patch_reads)
    npatches = indices_x.shape[0]
    patchwidth = 2 * rec.halfwidth + 1
    con_marray, cat_marray = None, None
    if feature_source.continuous:
        con_data_cache = _get_rows(patch_data_slices,
                                   feature_source.continuous)
        con_marray = _cached_read(con_data_cache,
                                  feature_source.continuous,
                                  patch_reads, mask_reads, npatches,
                                  patchwidth)
    if feature_source.categorical:
        cat_data_cache = _get_rows(patch_data_slices,
                                   feature_source.categorical)
        cat_marray = _cached_read(cat_data_cache,
                                  feature_source.categorical,
                                  patch_reads, mask_reads, npatches,
                                  patchwidth)
    coords = np.vstack((coords_x, coords_y)).T
    output = DataArrays(con_marray, cat_marray, None, coords, indices)
    return output



class TrainingDataProcessor(Worker):

    def __init__(self, tinfo: SourceMetadata) -> None:
        self.source_info = tinfo
        self.feature_source: Optional[H5Features] = None

    def __call__(self, values: Tuple[np.ndarray, np.ndarray]) -> List[bytes]:
        if not self.feature_source:
            self.feature_source = H5Features(self.source_info.feature_path)
        targets, coords = values
        arrays = _process_training(coords, targets, self.feature_source,
                                   self.source_info)
        strings = serialise(arrays)
        return strings


class QueryDataProcessor(Worker):

    def __init__(self, qinfo: SourceMetadata) -> None:
        self.source_info = qinfo
        self.feature_source: Optional[H5Features] = None


    def __call__(self, indices: np.ndarray) -> List[bytes]:
        if not self.feature_source:
            self.feature_source = H5Features(self.source_info.feature_path)
        arrays = _process_query(indices, self.feature_source, self.source_info)
        strings = serialise(arrays)
        return strings
