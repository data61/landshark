"""write training data"""
from types import TracebackType
from itertools import groupby, count
import logging
import os.path

import numpy as np
from typing import List, Tuple, Dict, Iterator, cast, Optional, TypeVar, \
    NamedTuple
import tables


from landshark import patch
from landshark.multiproc import task_list
from landshark.basetypes import FixedSlice, Worker, IdReader, ArraySource
from landshark.patch import PatchRowRW, PatchMaskRowRW
from landshark.iteration import batch_slices
from landshark import tfwrite
from landshark.hread import H5Features, CategoricalH5ArraySource, \
    OrdinalH5ArraySource
from landshark.image import indices_strip, world_to_image, ImageSpec
from landshark.hread import read_image_spec
from landshark.serialise import serialise
from landshark.kfold import KFolds
from landshark.metadata import CategoricalMetadata, FeatureSetMetadata, \
    TargetMetadata

log = logging.getLogger(__name__)


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
                      feature_source: H5Features, image_spec: ImageSpec,
                      halfwidth: int) -> \
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
    ord_marray, cat_marray = None, None
    if feature_source.ordinal:
        ord_marray = _direct_read(feature_source.ordinal,
                                  patch_reads, mask_reads,
                                  npatches, patchwidth)
    if feature_source.categorical:
        cat_marray = _direct_read(feature_source.categorical,
                                  patch_reads, mask_reads,
                                  npatches, patchwidth)
    return ord_marray, cat_marray


def _process_query(indices: Tuple[np.ndarray, np.ndarray],
                   feature_source: H5Features,
                   image_spec: ImageSpec, halfwidth: int) \
        -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    indices_x, indices_y = indices
    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            halfwidth,
                                            image_spec.width,
                                            image_spec.height)
    patch_data_slices = _slices_from_patches(patch_reads)
    npatches = indices_x.shape[0]
    patchwidth = 2 * halfwidth + 1
    ord_marray, cat_marray = None, None
    if feature_source.ordinal:
        ord_data_cache = _get_rows(patch_data_slices,
                                   feature_source.ordinal)
        ord_marray = _cached_read(ord_data_cache,
                                  feature_source.ordinal,
                                  patch_reads, mask_reads, npatches,
                                  patchwidth)
    if feature_source.categorical:
        cat_data_cache = _get_rows(patch_data_slices,
                                   feature_source.categorical)
        cat_marray = _cached_read(cat_data_cache,
                                  feature_source.categorical,
                                  patch_reads, mask_reads, npatches,
                                  patchwidth)
    return ord_marray, cat_marray


class TrainingDataProcessor(Worker):

    def __init__(self, image_spec: ImageSpec, feature_path: str,
                 halfwidth: int) -> None:
        self.feature_path = feature_path
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source: Optional[H5Features] = None

    def __call__(self, values: Tuple[np.ndarray, np.ndarray]) -> \
            Tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ndarray]:
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        targets, coords = values
        ord_marray, cat_marray = _process_training(coords, self.feature_source,
                                                   self.image_spec,
                                                   self.halfwidth)
        return ord_marray, cat_marray, targets


class SerialisingTrainingDataProcessor(Worker):

    def __init__(self, image_spec: ImageSpec, feature_path:str,
                 halfwidth: int) -> None:
        self.proc = TrainingDataProcessor(image_spec, feature_path, halfwidth)

    def __call__(self, values: Tuple[np.ndarray, np.ndarray]) -> \
            List[bytes]:
        ord_marray, cat_marray, targets = self.proc(values)
        strings = serialise(ord_marray, cat_marray, targets)
        return strings


class QueryDataProcessor(Worker):

    def __init__(self, image_spec: ImageSpec, feature_path: str,
                 halfwidth: int) -> None:
        self.feature_path = feature_path
        self.halfwidth = halfwidth
        self.image_spec = image_spec
        self.feature_source: Optional[H5Features] = None

    def __call__(self, indices: Tuple[np.ndarray, np.ndarray]) -> \
            Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
        if not self.feature_source:
            self.feature_source = H5Features(self.feature_path)
        ord_marray, cat_marray = _process_query(indices, self.feature_source,
                                                self.image_spec,
                                                self.halfwidth)
        return ord_marray, cat_marray


class SerialisingQueryDataProcessor(Worker):

    def __init__(self, image_spec: ImageSpec, feature_path: str,
                 halfwidth: int) -> None:
        self.proc = QueryDataProcessor(image_spec, feature_path, halfwidth)

    def __call__(self, indices: Tuple[np.ndarray, np.ndarray]) -> \
            List[bytes]:
        ord_marray, cat_marray = self.proc(indices)
        strings = serialise(ord_marray, cat_marray, None)
        return strings


class SourceMetadata(NamedTuple):
    name: str
    feature_path: str
    target_src: ArraySource
    image_spec: ImageSpec
    halfwidth: int
    folds: KFolds

def setup_training(feature_path: str, feature_meta: FeatureSetMetadata,
                   target_path: str, target_meta: TargetMetadata,
                   folds: int, random_seed: int,
                   halfwidth: int) \
        -> SourceMetadata:
    name = os.path.basename(feature_path).rsplit("_features.")[0] + "-" + \
        os.path.basename(target_path).rsplit("_targets.")[0]

    target_src = CategoricalH5ArraySource(target_path) \
        if isinstance(target_meta, CategoricalMetadata) \
            else OrdinalH5ArraySource(target_path)

    n_rows = len(target_src)
    kfolds = KFolds(n_rows, folds, random_seed)
    result = SourceMetadata(name, feature_path, target_src,
                            feature_meta.image, halfwidth, kfolds)
    return result


def write_trainingdata(tinfo: SourceMetadata,
                       output_directory: str,
                       testfold: int,
                       batchsize: int,
                       nworkers: int
                       ) -> None:

    log.info("Testing data is fold {} of {}".format(testfold, tinfo.folds))
    log.info("Writing training data to tfrecord in {}-point batches".format(
        batchsize))
    n_rows = len(tinfo.target_src)
    worker = SerialisingTrainingDataProcessor(tinfo.image_spec,
                                              tinfo.feature_path,
                                              tinfo.halfwidth)
    tasks = list(batch_slices(batchsize, n_rows))
    out_it = task_list(tasks, tinfo.target_src, worker, nworkers)
    fold_it = tinfo.folds.iterator(batchsize)
    tfwrite.training(out_it, n_rows, output_directory, testfold, fold_it)


def write_querydata(feature_path: str,
                    image_spec: ImageSpec,
                    strip: int,
                    total_strips: int,
                    batchsize: int,
                    halfwidth: int,
                    n_workers: int,
                    output_directory: str,
                    tag: str) -> None:
    true_batchsize = batchsize * image_spec.width
    log.info("Writing query data to tfrecord in {}-row batches".format(
        batchsize))
    reader_src = IdReader()
    it, n_total = indices_strip(image_spec, strip, total_strips,
                                true_batchsize)
    worker = SerialisingQueryDataProcessor(image_spec, feature_path, halfwidth)
    tasks = list(it)
    out_it = task_list(tasks, reader_src, worker, n_workers)
    tfwrite.query(out_it, n_total, output_directory, tag)
