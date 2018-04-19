"""Importing routines for tif data."""
import logging

import numpy as np

import tables
from typing import List, Iterator, TypeVar, Optional, Tuple

from landshark.basetypes import (ArraySource, OrdinalArraySource,
                                 CategoricalArraySource, CoordinateArraySource,
                                 Worker, IdWorker)
from landshark.image import ImageSpec
from landshark.iteration import batch_slices, with_slices
from landshark.multiproc import task_list
from landshark.category import CategoryMapper, CategoryInfo
from landshark.normalise import Normaliser

log = logging.getLogger(__name__)


T = TypeVar("T")


def _cat(it: Iterator[Iterator[T]]) -> List[T]:
    result = [i for k in it for i in k]
    return result


def write_imagespec(spec: ImageSpec, hfile: tables.File) -> None:
    hfile.root._v_attrs.crs = spec.crs
    hfile.create_array(hfile.root, name="x_coordinates",
                       obj=spec.x_coordinates)
    hfile.create_array(hfile.root, name="y_coordinates",
                       obj=spec.y_coordinates)


def _write_stats(stats: Optional[Tuple[np.ndarray, np.ndarray]],
                 hfile: tables.File) -> None:
    if stats is not None:
        mean, variance = stats
    else:
        mean, variance = None, None
    hfile.root.ordinal_data.attrs.mean = mean
    hfile.root.ordinal_data.attrs.variance = variance


def _write_maps(maps: Optional[CategoryInfo],
                hfile: tables.File) -> None:
    if maps is not None:
        _make_int_vlarray(hfile, "categorical_mappings", maps.mappings)
        _make_int_vlarray(hfile, "categorical_counts", maps.counts)


def write_ordinal(source: OrdinalArraySource,
                  hfile: tables.File,
                  n_workers: int,
                  batchsize: Optional[int]=None,
                  stats: Optional[Tuple[np.ndarray, np.ndarray]]=None) \
        -> None:
    transform = Normaliser(*stats) if stats else IdWorker()
    n_workers = n_workers if stats else 0
    _write_source(source, hfile, tables.Float32Atom(source.shape[-1]),
                  "ordinal_data", transform, n_workers, batchsize)
    _write_stats(stats, hfile)


def write_categorical(source: CategoricalArraySource,
                      hfile: tables.File,
                      n_workers: int,
                      batchsize: Optional[int]=None,
                      maps: Optional[CategoryInfo]=None) -> None:
    transform = CategoryMapper(maps.mappings, source.missing) \
        if maps else IdWorker()
    n_workers = n_workers if maps else 0
    _write_source(source, hfile, tables.Int32Atom(source.shape[-1]),
                  "categorical_data", transform, n_workers, batchsize)
    _write_maps(maps, hfile)


def _write_source(src: ArraySource,
                  hfile: tables.File,
                  atom: tables.Atom,
                  name: str,
                  transform: Worker,
                  n_workers: int,
                  batchsize: Optional[int]=None) -> None:
    front_shape = src.shape[0:-1]
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = hfile.create_carray(hfile.root, name=name,
                                atom=atom, shape=front_shape, filters=filters)
    _make_str_vlarray(hfile, name + "_columns", src.columns)
    array.attrs.missing = src.missing

    batchsize = batchsize if batchsize else src.native
    log.info("Writing {} to HDF5 in {}-row batches".format(name, batchsize))
    _write(src, array, batchsize, n_workers, transform)


def _write(source: ArraySource, array: tables.CArray,
           batchsize: int, n_workers: int, transform: Worker) -> None:
    n_rows = len(source)
    slices = list(batch_slices(batchsize, n_rows))
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
