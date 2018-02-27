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

log = logging.getLogger(__name__)


def _cat(it):
    result = [i for k in it for i in k]
    return result

def write_imagespec(spec, h5file):
    h5file.root._v_attrs.crs = spec.crs
    h5file.create_array(h5file.root, name="x_coordinates",
                        obj=spec.x_coordinates)
    h5file.create_array(h5file.root, name="y_coordinates",
                        obj=spec.y_coordinates)


def write_stats(hfile, stats):
    mean, variance = stats
    hfile.root.ordinal_data.attrs.mean = mean
    hfile.root.ordinal_data.attrs.variance = variance


def write_maps(hfile, maps):
    _make_int_vlarray(hfile, "categorical_mappings", maps.mappings)
    _make_int_vlarray(hfile, "categorical_counts", maps.counts)
    hfile.root.categorical_data.attrs.missing = maps.missing


def write_ordinal(source, hfile, batchsize=None):
    _write_source(source, hfile, tables.Float32Atom(source.shape[-1]),
                  "ordinal_data", batchsize)

def write_categorical(source, hfile, batchsize=None):
    _write_source(source, hfile, tables.Int32Atom(source.shape[-1]),
                  "categorical_data", batchsize)


def _write_source(src, hfile, atom, name, batchsize=None):
    front_shape = src.shape[0:-1]
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = hfile.create_carray(hfile.root, name=name,
                                atom=atom, shape=front_shape, filters=filters)
    array.attrs.columns = src.columns
    array.attrs.missing = src.missing
    batchsize = batchsize if batchsize else src.native
    log.info("Writing {} to HDF5 in {}-row batches".format(name, batchsize))
    _write(src, array, batchsize)


def _write(source, array, batchsize):
    # Assume all the same
    n_rows = len(source)
    slices = list(batch_slices(batchsize, n_rows))
    with source:
        with tqdm(total=len(slices)) as pbar:
            for s in slices:
                array[s.start:s.stop] = source(s)
                array.flush()
                pbar.update()

def write_coordinates(array_src, h5file, batchsize):
    shape = array_src.shape[0:1]
    atom = tables.Float64Atom(shape=(array_src.shape[1],))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="coordinates",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = array_src.columns
    array.attrs.missing = array_src.missing
    it = batch_slices(batchsize, array_src.shape[0])
    for start_idx, stop_idx in it:
        array[start_idx: stop_idx] = array_src(FixedSlice(start_idx, stop_idx))

def _make_int_vlarray(h5file, name, attribute):
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Int32Atom(shape=()))
    for a in attribute:
        vlarray.append(a)
