"""Process training and query data."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from itertools import count, groupby
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import tables

from landshark import patch, tfwrite
from landshark.basetypes import ArraySource, FixedSlice, IdReader, Worker
from landshark.hread import H5Features
from landshark.image import (ImageSpec, image_to_world, indices_strip,
                             world_to_image)
from landshark.iteration import batch_slices
from landshark.kfold import KFolds
from landshark.multiproc import task_list
from landshark.patch import PatchMaskRowRW, PatchRowRW
from landshark.serialise import DataArrays, serialise

log = logging.getLogger(__name__)


class ProcessTrainingArgs(NamedTuple):
    name: str
    feature_path: str
    target_src: ArraySource
    image_spec: ImageSpec
    halfwidth: int
    testfold: int
    folds: KFolds
    directory: str
    batchsize: int
    nworkers: int


class ProcessQueryArgs(NamedTuple):
    name: str
    feature_path: str
    image_spec: ImageSpec
    strip_idx: int
    total_strips: int
    strip_spec: ImageSpec
    halfwidth: int
    directory: str
    batchsize: int
    nworkers: int
    tag: str


def _direct_read(array: tables.CArray,
                 patch_reads: List[PatchRowRW],
                 mask_reads: List[PatchMaskRowRW],
                 npatches: int,
                 patchwidth: int
                 ) -> np.ma.MaskedArray:
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
                 patchwidth: int
                 ) -> np.ma.MaskedArray:
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

    c_init = count()

    def _get(n: int, c: Iterator[int] = c_init) -> int:
        res = n - next(c)
        return res

    slices = [_as_range(g) for _, g in groupby(rowlist, key=_get)]
    return slices


def _get_rows(slices: List[FixedSlice],
              array: tables.CArray
              ) -> Dict[int, np.ndarray]:
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
                      image_spec: ImageSpec,
                      halfwidth: int
                      ) -> DataArrays:
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
                   image_spec: ImageSpec,
                   halfwidth: int
                   ) -> DataArrays:
    indices_x, indices_y = indices.T
    coords_x = image_to_world(indices_x, image_spec.x_coordinates)
    coords_y = image_to_world(indices_y, image_spec.y_coordinates)
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


class _TrainingDataProcessor(Worker):

    def __init__(self,
                 feature_path: str,
                 image_spec: ImageSpec,
                 halfwidth: int
                 ) -> None:
        self.feature_path = feature_path
        self.feature_source: Optional[H5Features] = None
        self.image_spec = image_spec
        self.halfwidth = halfwidth

    def __call__(self, values: Tuple[np.ndarray, np.ndarray]) -> List[bytes]:
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        targets, coords = values
        arrays = _process_training(coords, targets, self.feature_source,
                                   self.image_spec, self.halfwidth)
        strings = serialise(arrays)
        return strings


class _QueryDataProcessor(Worker):

    def __init__(self,
                 feature_path: str,
                 image_spec: ImageSpec,
                 halfwidth: int
                 ) -> None:
        self.feature_path = feature_path
        self.feature_source: Optional[H5Features] = None
        self.image_spec = image_spec
        self.halfwidth = halfwidth

    def __call__(self, indices: np.ndarray) -> List[bytes]:
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        arrays = _process_query(indices, self.feature_source, self.image_spec,
                                self.halfwidth)
        strings = serialise(arrays)
        return strings


def write_trainingdata(args: ProcessTrainingArgs) -> None:
    log.info("Testing data is fold {} of {}".format(args.testfold,
                                                    args.folds.K))
    log.info("Writing training data to tfrecord in {}-point batches".format(
        args.batchsize))
    n_rows = len(args.target_src)
    worker = _TrainingDataProcessor(args.feature_path, args.image_spec,
                                    args.halfwidth)
    tasks = list(batch_slices(args.batchsize, n_rows))
    out_it = task_list(tasks, args.target_src, worker, args.nworkers)
    fold_it = args.folds.iterator(args.batchsize)
    tfwrite.training(out_it, n_rows, args.directory, args.testfold, fold_it)


def write_querydata(args: ProcessQueryArgs) -> None:

    log.info("Query data is strip {} of {}".format(args.strip_idx,
                                                   args.total_strips))
    log.info("Writing query data to tfrecord in {}-point batches".format(
        args.batchsize))
    reader_src = IdReader()
    it, n_total = indices_strip(args.image_spec, args.strip_idx,
                                args.total_strips, args.batchsize)
    worker = _QueryDataProcessor(args.feature_path, args.strip_spec,
                                 args.halfwidth)
    tasks = list(it)
    out_it = task_list(tasks, reader_src, worker, args.nworkers)
    tfwrite.query(out_it, n_total, args.directory, args.tag)
