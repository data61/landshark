"""Importing routines for tif data."""
import logging

import tables
# from typing import List, Union, Callable, Iterator

from landshark.basetypes import get_metadata, ClassSpec, FixedSlice
from landshark.category import get_categories, CategoricalOutputTransform
from landshark.normalise import get_stats, OrdinalOutputTransform
from landshark.iteration import batch_slices, with_slices
from landshark.multiproc import task_list

log = logging.getLogger(__name__)


def write_imagespec(spec, h5file):
    h5file.root._v_attrs.crs = spec.crs
    h5file.create_array(h5file.root, name="x_coordinates",
                        obj=spec.x_coordinates)
    h5file.create_array(h5file.root, name="y_coordinates",
                        obj=spec.y_coordinates)


def write_ordinal(array_src_spec, h5file, batchsize, n_workers,
                  normalise=True):
    meta = get_metadata(array_src_spec)
    batchsize = batchsize if batchsize else meta.native
    shape = meta.shape[0:-1]
    # The bands will form part of the atom
    nbands = meta.shape[-1]
    atom = tables.Float32Atom(shape=(nbands,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="ordinal_data",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = meta.columns
    missing = meta.missing
    array.attrs.missing = missing
    mean, variance = None, None
    if normalise:
        log.info("Computing statistics for standardisation:")
        mean, variance = get_stats(array_src_spec, meta, batchsize, n_workers)
    array.attrs.mean = mean
    array.attrs.variance = variance

    worker_spec = ClassSpec(OrdinalOutputTransform, [mean, variance, missing])
    log.info("Writing ordinal data to disk:")
    _write(array_src_spec, meta, array, worker_spec, batchsize, n_workers)


def write_categorical(array_src_spec, h5file, batchsize, n_workers):
    meta = get_metadata(array_src_spec)
    shape = meta.shape[0:-1]
    # The bands will form part of the atom
    nbands = meta.shape[-1]
    atom = tables.Int32Atom(shape=(nbands,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="categorical_data",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = meta.columns
    res = get_categories(array_src_spec, meta, batchsize, n_workers)

    _make_int_vlarray(h5file, "categorical_mappings", res.mappings)
    _make_int_vlarray(h5file, "categorical_counts", res.counts)
    array.attrs.missing = res.missing

    worker_spec = ClassSpec(CategoricalOutputTransform, [res.mappings])
    log.info("Writing categorical data to disk:")
    _write(array_src_spec, meta, array, worker_spec, batchsize, n_workers)


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


def _write(source_spec, meta, array, worker_spec, batchsize, n_workers):
    n_rows = meta.shape[0]
    slices = list(batch_slices(batchsize, n_rows))
    out_it = task_list(slices, source_spec, worker_spec, n_workers)
    for (start_idx, end_idx), d in with_slices(out_it):
        array[start_idx:end_idx] = d
        array.flush()


def _make_int_vlarray(h5file, name, attribute):
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    vlarray = h5file.create_vlarray(h5file.root, name=name,
                                    atom=tables.Int32Atom(shape=()),
                                    filters=filters)
    for a in attribute:
        vlarray.append(a)
