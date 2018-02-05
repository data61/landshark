"""Importing routines for tif data."""
import logging

import tables
# from typing import List, Union, Callable, Iterator

# from landshark.importers.tifread import ImageStack
from landshark.category import get_categories, CategoricalOutputTransform
from landshark.normalise import get_stats, OrdinalOutputTransform
from landshark.iteration import batch_slices, with_slices

log = logging.getLogger(__name__)


def write_imagespec(spec, h5file):
    h5file.root._v_attrs.crs = spec.crs
    h5file.create_array(h5file.root, name="x_coordinates",
                        obj=spec.x_coordinates)
    h5file.create_array(h5file.root, name="y_coordinates",
                        obj=spec.y_coordinates)


def write_ordinal(array_src, h5file, batchsize, pool, normalise=True):
    batchsize = batchsize if batchsize else array_src.native
    shape = array_src.shape[0:-1]
    # The bands will form part of the atom
    nbands = array_src.shape[-1]
    atom = tables.Float32Atom(shape=(nbands,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="ordinal_data",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = array_src.columns
    missing = array_src.missing
    array.attrs.missing = missing

    mean, variance = None, None
    if normalise:
        log.info("Computing statistics for standardisation:")
        mean, variance = get_stats(array_src, batchsize, pool)
    array.attrs.mean = mean
    array.attrs.variance = variance

    f = OrdinalOutputTransform(mean, variance, missing)
    log.info("Writing ordinal data to disk:")
    _write(array_src, array, f, batchsize, pool)


def write_categorical(array_src, h5file, batchsize, pool):
    shape = array_src.shape[0:-1]
    # The bands will form part of the atom
    nbands = array_src.shape[-1]
    atom = tables.Int32Atom(shape=(nbands,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="categorical_data",
                                 atom=atom, shape=shape, filters=filters)

    array.attrs.columns = array_src.columns
    # We always map missing values to zero
    res = get_categories(array_src, batchsize, pool)
    mappings, counts, missing = res.mappings, res.counts, res.missing

    array.attrs.mappings = mappings
    array.attrs.counts = counts
    array.attrs.missing = missing

    f = CategoricalOutputTransform(mappings)
    log.info("Writing categorical data to disk:")
    _write(array_src, array, f, batchsize, pool)


def write_coordinates(array_src, h5file, batchsize):
    shape = array_src.shape[0:1]
    atom = tables.Float64Atom(shape=(array_src.shape[1],))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="coordinates",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = array_src.columns
    array.attrs.missing = array_src.missing
    it = batch_slices(batchsize, array_src.shape[0])
    for start_idx, end_idx in it:
        array[start_idx: end_idx] = array_src.slice(start_idx, end_idx)


def _write(source, array, f, batchsize, pool):
    n_rows = len(source)
    it = batch_slices(batchsize, n_rows)
    data_it = ((source.slice(start, end)) for start, end in it)
    out_it = pool.imap(f, data_it)
    for (start_idx, end_idx), d in with_slices(out_it):
        array[start_idx:end_idx] = d
        array.flush()
