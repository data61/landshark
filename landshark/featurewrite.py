"""Importing routines for tif data."""
import logging
from typing import Iterator, List, Optional, Tuple, TypeVar

import numpy as np
import tables

from landshark.basetypes import (ArraySource, CategoricalArraySource,
                                 ContinuousArraySource, CoordinateArraySource,
                                 IdWorker, Worker)
from landshark.category import CategoryMapper
from landshark.image import ImageSpec
from landshark.iteration import batch_slices, with_slices
from landshark.metadata import (CategoricalFeatureSet, CategoricalTarget,
                                ContinuousFeatureSet, ContinuousTarget,
                                FeatureSet, Target)
from landshark.multiproc import task_list
from landshark.normalise import Normaliser

log = logging.getLogger(__name__)


T = TypeVar("T")


def _cat(it: Iterator[Iterator[T]]) -> List[T]:
    result = [i for k in it for i in k]
    return result


def write_feature_metadata(meta: FeatureSet, hfile: tables.File) -> None:
    hfile.root._v_attrs.N = len(meta)
    hfile.root._v_attrs.halfwidth = meta.halfwidth
    write_imagespec(meta.image, hfile)
    if meta.continuous:
        _write_continuous_metadata(meta.continuous, hfile)
    if meta.categorical:
        _write_categorical_metadata(meta.categorical, hfile)


def read_feature_metadata(path: str) -> FeatureSet:
    with tables.open_file(path, 'r') as hfile:
        N = hfile.root._v_attrs.N
        halfwidth = hfile.root._v_attrs.halfwidth
        image_spec = read_imagespec(hfile)
        continuous, categorical = None, None
        if hasattr(hfile.root, "continuous_data"):
            continuous = _read_continuous_metadata(hfile)
        if hasattr(hfile.root, "categorical_data"):
            categorical = _read_categorical_metadata(hfile)
    m = FeatureSet(continuous, categorical, image_spec, N, halfwidth)
    return m


def write_target_metadata(meta: Target, hfile: tables.File) -> None:
    if isinstance(meta, ContinuousTarget):
        _write_continuous_target_metadata(meta, hfile)
    elif isinstance(meta, CategoricalTarget):
        _write_categorical_target_metadata(meta, hfile)
    else:
        raise RuntimeError("Don't recognise type of target metadata")


def read_target_metadata(path: str) -> Target:
    with tables.open_file(path, 'r') as hfile:
        if hasattr(hfile.root, "continuous_data"):
            continuous = _read_continuous_target_metadata(hfile)
            return continuous
        elif hasattr(hfile.root, "categorical_data"):
            categorical = _read_categorical_target_metadata(hfile)
            return categorical
        else:
            raise RuntimeError("Can't find Metadata")


def _write_continuous_metadata(meta: ContinuousFeatureSet,
                               hfile: tables.File
                               ) -> None:
    hfile.root.continuous_data.attrs.missing = meta.missing_value
    hfile.root.continuous_data.attrs.normalised = meta.normalised
    labels = [k for k in meta.columns.keys()]
    D = np.array([v.D for v in meta.columns.values()], dtype=int)
    means = [v.mean for v in meta.columns.values()]
    sds = [v.sd for v in meta.columns.values()]
    _make_str_vlarray(hfile, "continuous_labels", labels)
    hfile.create_array(hfile.root, name="continuous_D", obj=D)
    if meta.normalised:
        _make_float_vlarray(hfile, "continuous_means", means)
        _make_float_vlarray(hfile, "continuous_sds", sds)


def _read_continuous_metadata(hfile: tables.File) -> ContinuousFeatureSet:
    missing_value = hfile.root.continuous_data.attrs.missing
    normalised = hfile.root.continuous_data.attrs.normalised
    labels = [k.decode() for k in hfile.root.continuous_labels.read()]
    means, sds = None, None
    if normalised:
        means = hfile.root.continuous_means.read()
        sds = hfile.root.continuous_sds.read()
    meta = ContinuousFeatureSet(labels, missing_value, means, sds)
    return meta


def _write_continuous_target_metadata(meta: ContinuousTarget,
                                      hfile: tables.File
                                      ) -> None:
    hfile.root.continuous_data.attrs.D = meta.D
    hfile.root.continuous_data.attrs.N = meta.N
    hfile.root.continuous_data.attrs.normalised = meta.normalised
    _make_str_vlarray(hfile, "continuous_labels", meta.labels)
    if meta.normalised:
        _make_float_vlarray(hfile, "continuous_means", meta.means)
        _make_float_vlarray(hfile, "continuous_sds", meta.sds)


def _read_continuous_target_metadata(hfile: tables.File) -> ContinuousTarget:
    normalised = hfile.root.continuous_data.attrs.normalised
    N = hfile.root.continuous_data.attrs.N
    labels = [k.decode() for k in hfile.root.continuous_labels.read()]
    means, sds = None, None
    if normalised:
        means = hfile.root.continuous_means.read()
        sds = hfile.root.continuous_sds.read()
    meta = ContinuousTarget(N, labels, means, sds)
    return meta


def _write_categorical_metadata(meta: CategoricalFeatureSet,
                                hfile: tables.File
                                ) -> None:
    hfile.root.categorical_data.attrs.missing = meta.missing_value
    labels = [k for k in meta.columns.keys()]
    nvalues = np.array([v.nvalues for v in meta.columns.values()])
    D = np.array([v.D for v in meta.columns.values()])
    mappings = [v.mapping for v in meta.columns.values()]
    counts = [v.counts for v in meta.columns.values()]
    _make_str_vlarray(hfile, "categorical_labels", labels)
    hfile.create_array(hfile.root, name="categorical_D", obj=D)
    _make_int_vlarray(hfile, "categorical_counts", counts)
    _make_int_vlarray(hfile, "categorical_mappings", mappings)
    hfile.create_array(hfile.root, name="categorical_nvalues", obj=nvalues)


def _read_categorical_metadata(hfile: tables.File) -> CategoricalFeatureSet:
    missing_value = hfile.root.categorical_data.attrs.missing
    labels = [k.decode() for k in hfile.root.categorical_labels.read()]

    mappings = hfile.root.categorical_mappings.read()
    counts = hfile.root.categorical_counts.read()
    nvalues = np.array(hfile.root.categorical_nvalues.read())
    meta = CategoricalFeatureSet(labels, missing_value,
                                 nvalues, mappings, counts)
    return meta


def _write_categorical_target_metadata(meta: CategoricalTarget,
                                       hfile: tables.File
                                       ) -> None:
    hfile.root.categorical_data.attrs.D = meta.D
    hfile.root.categorical_data.attrs.N = meta.N
    _make_str_vlarray(hfile, "categorical_labels", meta.labels)
    _make_int_vlarray(hfile, "categorical_counts", meta.counts)
    _make_int_vlarray(hfile, "categorical_mappings", meta.mappings)
    hfile.create_array(hfile.root, name="categorical_nvalues",
                       obj=meta.nvalues)


def _read_categorical_target_metadata(hfile: tables.File) -> CategoricalTarget:
    N = hfile.root.categorical_data.attrs.N
    labels = [k.decode() for k in hfile.root.categorical_labels.read()]
    mappings = hfile.root.categorical_mappings.read()
    counts = hfile.root.categorical_counts.read()
    nvalues = hfile.root.categorical_nvalues.read()

    meta = CategoricalTarget(N, labels, nvalues, mappings, counts)
    return meta


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


def write_continuous(source: ContinuousArraySource,
                     hfile: tables.File,
                     n_workers: int,
                     batchrows: Optional[int] = None,
                     stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
                     ) -> None:
    transform = Normaliser(*stats, source.missing) if stats else IdWorker()
    n_workers = n_workers if stats else 0
    _write_source(source, hfile, tables.Float32Atom(source.shape[-1]),
                  "continuous_data", transform, n_workers, batchrows)


def write_categorical(source: CategoricalArraySource,
                      hfile: tables.File,
                      n_workers: int,
                      batchrows: Optional[int] = None,
                      maps: Optional[np.ndarray] = None
                      ) -> None:
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
                  batchrows: Optional[int] = None
                  ) -> None:
    front_shape = src.shape[0:-1]
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = hfile.create_carray(hfile.root, name=name,
                                atom=atom, shape=front_shape, filters=filters)
    array.attrs.missing = src.missing
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
                      h5file: tables.File,
                      batchsize: int
                      ) -> None:
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


def _make_int_vlarray(h5file: tables.File,
                      name: str,
                      attribute: np.ndarray
                      ) -> None:
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Int32Atom(shape=()))
    for a in attribute:
        vlarray.append(a)


def _make_float_vlarray(h5file: tables.File,
                        name: str,
                        attribute: np.ndarray
                        ) -> None:
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Float64Atom(shape=()))
    for a in attribute:
        vlarray.append(a)


def _make_str_vlarray(h5file: tables.File,
                      name: str,
                      attribute: List[str]
                      ) -> None:
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.VLStringAtom())
    for a in attribute:
        vlarray.append(a)
