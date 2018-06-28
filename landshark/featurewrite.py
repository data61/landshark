"""Importing routines for tif data."""
import logging

import numpy as np

import tables
from typing import List, Iterator, TypeVar, Optional, Tuple, Union, cast

from landshark.basetypes import (ArraySource, OrdinalArraySource,
                                 CategoricalArraySource, CoordinateArraySource,
                                 Worker, IdWorker)
from landshark.image import ImageSpec
from landshark.iteration import batch_slices, with_slices
from landshark.multiproc import task_list
from landshark.category import CategoryMapper, CategoryInfo
from landshark.normalise import Normaliser
from landshark.metadata import OrdinalMetadata, CategoricalMetadata, \
    FeatureSetMetadata, TargetMetadata

log = logging.getLogger(__name__)


T = TypeVar("T")


def _cat(it: Iterator[Iterator[T]]) -> List[T]:
    result = [i for k in it for i in k]
    return result


def write_feature_metadata(meta: FeatureSetMetadata,
                           hfile: tables.File) -> None:
    hfile.root._v_attrs.N = meta.N
    write_imagespec(meta.image, hfile)
    if meta.ordinal:
        write_ordinal_metadata(meta.ordinal, hfile)
    if meta.categorical:
        write_categorical_metadata(meta.categorical, hfile)

def read_featureset_metadata(path: str) -> FeatureSetMetadata:
    with tables.open_file(path, 'r') as hfile:
        N = hfile.root._v_attrs.N
        image_spec = read_imagespec(hfile)
        ordinal, categorical = None, None
        if hasattr(hfile.root, "ordinal_data"):
            ordinal = read_ordinal_metadata(hfile)
        if hasattr(hfile.root, "categorical_data"):
            categorical = read_categorical_metadata(hfile)
    m = FeatureSetMetadata(ordinal, categorical, image_spec)
    return m

def read_target_metadata(path: str) -> TargetMetadata:
    with tables.open_file(path, 'r') as hfile:
        if hasattr(hfile.root, "ordinal_data"):
            ord_meta = read_ordinal_metadata(hfile)
            return ord_meta
        elif hasattr(hfile.root, "categorical_data"):
            cat_meta = read_categorical_metadata(hfile)
            return cat_meta
        else:
            raise RuntimeError("Can't find data in target file")

def write_ordinal_metadata(meta: OrdinalMetadata,
                           hfile: tables.File) -> None:
    hfile.root._v_attrs.ordinal_N = meta.N
    hfile.root.ordinal_data.attrs.missing = meta.missing
    hfile.root.ordinal_data.attrs.D = meta.D
    _make_str_vlarray(hfile, "ordinal_labels", meta.labels)
    hfile.root.ordinal_data.attrs.mean = meta.means
    hfile.root.ordinal_data.attrs.variance = meta.variances

def read_ordinal_metadata(hfile: tables.File) -> OrdinalMetadata:
    N = hfile.root._v_attrs.ordinal_N
    missing = hfile.root.ordinal_data.attrs.missing
    D = hfile.root.ordinal_data.attrs.D
    labels = [k.decode() for k in hfile.root.ordinal_labels.read()]
    mean = hfile.root.ordinal_data.attrs.mean
    var = hfile.root.ordinal_data.attrs.variance
    m = OrdinalMetadata(N, D, labels, missing, mean, var)
    return m

def write_categorical_metadata(meta: CategoricalMetadata,
                               hfile: tables.File) -> None:

    hfile.root._v_attrs.categorical_N = meta.N
    hfile.root.categorical_data.attrs.missing = meta.missing
    hfile.root.categorical_data.attrs.D = meta.D
    _make_str_vlarray(hfile, "categorical_labels", meta.labels)
    hfile.create_array(hfile.root, name="ncategories",
                       obj=meta.ncategories)
    _make_int_vlarray(hfile, "categorical_counts", meta.counts)
    _make_int_vlarray(hfile, "categorical_mappings", meta.mappings)

def read_categorical_metadata(hfile: tables.File) -> CategoricalMetadata:
    N = hfile.root._v_attrs.categorical_N
    missing = hfile.root.categorical_data.attrs.missing
    D = hfile.root.categorical_data.attrs.D
    labels = [k.decode() for k in hfile.root.categorical_labels.read()]
    ncats = hfile.root.ncategories.read()
    mappings = hfile.root.categorical_mappings.read()
    counts = hfile.root.categorical_counts.read()
    m = CategoricalMetadata(N, D, labels, missing, ncats, mappings, counts)
    return m


def write_imagespec(spec: ImageSpec, hfile: tables.File) -> None:
    hfile.root._v_attrs.crs = spec.crs
    hfile.create_array(hfile.root, name="x_coordinates",
                       obj=spec.x_coordinates)
    hfile.create_array(hfile.root, name="y_coordinates",
                       obj=spec.y_coordinates)


def read_imagespec(hfile: tables.File) -> ImageSpec:
    crs = hfile.root._v_attrs.crs
    x_coordinates = np.array(hfile.root.x_coordinates)
    y_coordinates = np.array(hfile.root.y_coordinates)
    imspec = ImageSpec(x_coordinates, y_coordinates, crs)
    return imspec

def write_ordinal(source: OrdinalArraySource,
                  hfile: tables.File,
                  n_workers: int,
                  batchrows: Optional[int]=None,
                  stats: Optional[Tuple[np.ndarray, np.ndarray]]=None) \
        -> None:
    transform = Normaliser(*stats, source.missing) if stats else IdWorker()
    n_workers = n_workers if stats else 0
    _write_source(source, hfile, tables.Float32Atom(source.shape[-1]),
                  "ordinal_data", transform, n_workers, batchrows)


def write_categorical(source: CategoricalArraySource,
                      hfile: tables.File,
                      n_workers: int,
                      batchrows: Optional[int]=None,
                      maps: Optional[np.ndarray]=None) -> None:
    transform = CategoryMapper(maps, source.missing) \
        if maps else IdWorker()
    n_workers = n_workers if maps else 0
    _write_source(source, hfile, tables.Int32Atom(source.shape[-1]),
                  "categorical_data", transform, n_workers, batchrows)


def _write_source(src: ArraySource,
                  hfile: tables.File,
                  atom: tables.Atom,
                  name: str,
                  transform: Worker,
                  n_workers: int,
                  batchrows: Optional[int]=None) -> None:
    front_shape = src.shape[0:-1]
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = hfile.create_carray(hfile.root, name=name,
                                atom=atom, shape=front_shape, filters=filters)
    batchrows = batchrows if batchrows else src.native
    log.info("Writing {} to HDF5 in {}-row batches".format(name, batchrows))
    _write(src, array, batchrows, n_workers, transform)


def _write(source: ArraySource, array: tables.CArray,
           batchrows: int, n_workers: int, transform: Worker) -> None:
    n_rows = len(source)
    slices = list(batch_slices(batchrows, n_rows))
    out_it = task_list(slices, source, transform, n_workers)
    for s, d in with_slices(out_it):
        array[s.start:s.stop] = d
    array.flush()


def write_coordinates(array_src: CoordinateArraySource,
                      h5file: tables.File, batchsize: int) -> None:
    with array_src:
        shape = array_src.shape[0:1]
        atom = tables.Float64Atom(shape=(array_src.shape[1],))
        filters = tables.Filters(complevel=1, complib="blosc:lz4")
        array = h5file.create_carray(h5file.root, name="coordinates",
                                     atom=atom, shape=shape, filters=filters)
        _make_str_vlarray(h5file, "coordinates_columns", array_src.columns)
        array.attrs.missing = array_src.missing
        for s in batch_slices(batchsize, array_src.shape[0]):
            array[s.start:s.stop] = array_src(s)


def _make_int_vlarray(h5file: tables.File, name: str,
                      attribute: np.ndarray) -> None:
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Int32Atom(shape=()))
    for a in attribute:
        vlarray.append(a)


def _make_str_vlarray(h5file: tables.File, name: str,
                      attribute: List[str]) -> None:
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.VLStringAtom())
    for a in attribute:
        vlarray.append(a)
