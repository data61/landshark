"""Importing routines for tif data."""
import logging
from operator import iadd
from functools import reduce
from contextlib import ExitStack

from tqdm import tqdm
import tables
# from typing import List, Union, Callable, Iterator

from landshark.basetypes import FixedSlice
from landshark.iteration import batch_slices, with_slices
from landshark.multiproc import task_list
from landshark.category import CategoryMapper
from landshark.normalise import Normaliser

log = logging.getLogger(__name__)


def _id(x):
    return x

def _cat(it):
    result = [i for k in it for i in k]
    return result


def write_imagespec(spec, h5file):
    h5file.root._v_attrs.crs = spec.crs
    h5file.create_array(h5file.root, name="x_coordinates",
                        obj=spec.x_coordinates)
    h5file.create_array(h5file.root, name="y_coordinates",
                        obj=spec.y_coordinates)


def _write_stats(hfile, stats):
    if stats is not None:
        mean, variance = stats
    else:
        mean, variance = None, None
    hfile.root.ordinal_data.attrs.mean = mean
    hfile.root.ordinal_data.attrs.variance = variance


def _write_maps(hfile, maps):
    if maps is not None:
        _make_int_vlarray(hfile, "categorical_mappings", maps.mappings)
        _make_int_vlarray(hfile, "categorical_counts", maps.counts)


def write_ordinal(source, hfile, n_workers, batchsize=None, stats=None):
    transform = Normaliser(*stats) if stats else _id
    n_workers = n_workers if stats else 0
    _write_source(source, hfile, tables.Float32Atom(source.shape[-1]),
                  "ordinal_data", transform, n_workers, batchsize)
    _write_stats(hfile, stats)

def write_categorical(source, hfile, n_workers, batchsize=None, maps=None):
    transform = CategoryMapper(maps.mappings, source.missing) if maps else _id
    n_workers = n_workers if maps else 0
    _write_source(source, hfile, tables.Int32Atom(source.shape[-1]),
                  "categorical_data", transform, n_workers, batchsize)
    _write_maps(hfile, maps)


def _write_source(src, hfile, atom, name, transform,
                  n_workers, batchsize=None):
    front_shape = src.shape[0:-1]
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = hfile.create_carray(hfile.root, name=name,
                                atom=atom, shape=front_shape, filters=filters)
    array.attrs.columns = src.columns
    array.attrs.missing = src.missing
    batchsize = batchsize if batchsize else src.native
    log.info("Writing {} to HDF5 in {}-row batches".format(name, batchsize))
    _write(src, array, batchsize, n_workers, transform)


def _write(source, array, batchsize, n_workers, transform):
    # Assume all the same
    n_rows = len(source)
    slices = list(batch_slices(batchsize, n_rows))
    out_it = task_list(slices, source, transform, n_workers)
    for s, d in with_slices(out_it):
        array[s.start:s.stop] = d
    array.flush()

def write_coordinates(array_src, h5file, batchsize):
    with array_src:
        shape = array_src.shape[0:1]
        atom = tables.Float64Atom(shape=(array_src.shape[1],))
        filters = tables.Filters(complevel=1, complib="blosc:lz4")
        array = h5file.create_carray(h5file.root, name="coordinates",
                                     atom=atom, shape=shape, filters=filters)
        array.attrs.columns = array_src.columns
        array.attrs.missing = array_src.missing
        for s in batch_slices(batchsize, array_src.shape[0]):
            array[s.start:s.stop] = array_src(s)


def _make_int_vlarray(h5file, name, attribute):
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Int32Atom(shape=()))
    for a in attribute:
        vlarray.append(a)
