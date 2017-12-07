"""Importing routines for tif data."""

import os.path
import logging

import numpy as np
import tables
from typing import List, Union, Callable, Iterator

# from landshark.importers.tifread import ImageStack
from landshark.util import to_masked
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


def write_ordinal(source, h5file, batchsize, pool):
    array_src = source.ordinal
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

    log.info("Computing statistics for standardisation:")
    mean, variance = get_stats(source, batchsize, pool)
    array.attrs.mean = mean
    array.attrs.variance = variance

    f = OrdinalOutputTransform(mean, variance, missing)
    log.info("Writing ordinal data to disk:")
    _write(source, array, f, batchsize, pool)


def write_categorical(source, h5file, batchsize, pool):
    array_src = source.categorical
    shape = array_src.shape[0:-1]
    # The bands will form part of the atom
    nbands = array_src.shape[-1]
    atom = tables.Int32Atom(shape=(nbands,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="categorical_data",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = array_src.columns
    array.attrs.missing = array_src.missing

    mappings, counts = get_categories(source, batchsize, pool)
    array.attrs.mappings = mappings
    array.attrs.counts = counts

    f = CategoricalOutputTransform(mappings)
    log.info("Writing categorical data to disk:")
    _write(source, array, f, batchsize, pool)


def write_pointspec(source, h5file, batchsize):
    shape = source.ordinal.shape[0:1]
    atom = tables.Float64Atom(shape=(source.ordinal.shape[1],))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    array = h5file.create_carray(h5file.root, name="coordinates",
                                 atom=atom, shape=shape, filters=filters)
    array.attrs.columns = source.ordinal.columns
    array.attrs.missing = source.ordinal.missing
    it = batch_slices(batchsize, source.ordinal.shape[0])
    for start_idx, end_idx in it:
        array[start_idx: end_idx] = source.slice(start_idx, end_idx).ordinal


def _write(source, array, f, batchsize, pool):
    n_rows = len(source)
    it = batch_slices(batchsize, n_rows)
    data_it = ((source.slice(start, end)) for start, end in it)
    out_it = pool.imap(f, data_it)
    for (start_idx, end_idx), d in with_slices(out_it):
        array[start_idx:end_idx] = d
