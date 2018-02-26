"""Importing routines for tif data."""
import logging
from operator import iadd
from functools import reduce

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
    pass
    _make_int_vlarray(hfile, "categorical_mappings", maps.mappings)
    _make_int_vlarray(hfile, "categorical_counts", maps.counts)
    hfile.root.categorical_data.attrs.missing = maps.missing

def write_ordinal(sources, hfile, batchsize):
    _write_sources(sources, hfile, batchsize, tables.Float32Atom(),
                   "ordinal_data")

def write_categorical(sources, hfile, batchsize):
    _write_sources(sources, hfile, batchsize, tables.Int32Atom(),
                   "categorical_data")


def _write_sources(sources, hfile, batchsize, atom, name):
    total_features = sum(s.shape[-1] for s in sources)
    front_shape = sources[0].shape[0:-1]
    shape = tuple(list(front_shape) + [total_features])
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = hfile.create_carray(hfile.root, name=name,
                                atom=atom, shape=shape, filters=filters)
    array.attrs.columns = _cat(k.columns for k in sources)

    # check missingness
    if len({k.missing for k in sources}) > 1:
        raise ValueError("Sources do not agree on missing value mask")
    array.attrs.missing = sources[0].missing
    _write(sources, array, batchsize)


def _write(sources, array, batchsize):
    start_band = 0
    for src in sources:
        batchsize = batchsize if batchsize else src._native
        log.info("Writing {} to HDF5".format(src.name))
        stop_band = start_band + src.shape[-1]
        n_rows = len(src)
        slices = list(batch_slices(batchsize, n_rows))
        with src:
            with tqdm(total=len(slices)) as pbar:
                for s in slices:
                    array[s.start:s.stop, ..., start_band:stop_band] = src(s)
                    array.flush()
                    pbar.update()
        start_band = stop_band


# def write_coordinates(array_src, h5file, batchsize):
#     shape = array_src.shape[0:1]
#     atom = tables.Float64Atom(shape=(array_src.shape[1],))
#     filters = tables.Filters(complevel=1, complib="blosc:lz4")
#     array = h5file.create_carray(h5file.root, name="coordinates",
#                                  atom=atom, shape=shape, filters=filters)
#     array.attrs.columns = array_src.columns
#     array.attrs.missing = array_src.missing
#     it = batch_slices(batchsize, array_src.shape[0])
#     for start_idx, stop_idx in it:
#         array[start_idx: stop_idx] = array_src(FixedSlice(start_idx, stop_idx))

def _make_int_vlarray(h5file, name, attribute):
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Int32Atom(shape=()))
    for a in attribute:
        vlarray.append(a)
