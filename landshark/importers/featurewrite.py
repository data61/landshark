"""Importing routines for tif data."""

import os.path
import logging

import numpy as np
import tables
from typing import List, Union, Callable, Iterator

from landshark.importers.tifread import ImageStack
from landshark.importers.category import _Categories
from landshark.importers.normalise import _Statistics

log = logging.getLogger(__name__)

MissingValueList = List[Union[np.float32, np.int32, None]]


def write_datafile(image_stack: ImageStack, filename: str,
                   standardise: bool) -> None:
    """
    Write an ImageStack object to an HDF5 representation on disk.

    This function assumes writes iteratively from the image_stack,
    and therefore should support extremely large files.

    Parameters
    ----------
    image_stack : ImageStack
        The stack to write out (incrementally, need not fit on disk)
    filename : str
        The filename of the output HDF5 file.
    standardise : bool
        If true, rescale each ordinal feature to have mean 0 and std 1.

    """
    title = "Landshark Image Stack"
    log.info("Creating HDF5 output file")
    h5file = tables.open_file(filename, mode="w", title=title)

    # write the attributes to root
    log.info("Writing global attributes")
    attributes = h5file.root._v_attrs
    attributes.height = image_stack.height
    attributes.width = image_stack.width
    attributes.affine = image_stack.affine
    attributes.crs = image_stack.crs
    coords_x = image_stack.coordinates_x
    coords_y = image_stack.coordinates_y
    h5file.create_array(h5file.root, name="x_coordinates", obj=coords_x)
    h5file.create_array(h5file.root, name="y_coordinates", obj=coords_y)

    nbands_cat = len(image_stack.categorical_bands)
    nbands_ord = len(image_stack.ordinal_bands)
    cat_atom = tables.Int32Atom(shape=(nbands_cat,))
    ord_atom = tables.Float32Atom(shape=(nbands_ord,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")

    log.info("Creating data arrays")
    im_shape = (image_stack.height, image_stack.width)
    cat_array = h5file.create_carray(h5file.root, name="categorical_data",
                                     atom=cat_atom, shape=im_shape,
                                     filters=filters)
    cat_array.attrs.labels = image_stack.categorical_names
    ord_array = h5file.create_carray(h5file.root, name="ordinal_data",
                                     atom=ord_atom, shape=im_shape,
                                     filters=filters)
    ord_array.attrs.labels = image_stack.ordinal_names
    ord_array.attrs.missing_values = image_stack.ordinal_missing
    log.info("Categorical HDF5 block shape: {}".format(cat_array.chunkshape))
    log.info("Ordinal HDF5 block shape: {}".format(ord_array.chunkshape))

    log.info("Transforming and writing categorical data")
    categories = _Categories(image_stack.categorical_missing)
    _categorical_write(cat_array, image_stack.categorical_blocks, categories)
    cat_array.attrs.mappings = categories.maps
    cat_array.attrs.ncategories = [len(k) for k in categories.maps]
    cat_array.attrs.missing_values = categories.missing_values

    if standardise:
        stats = _Statistics(nbands_ord)
        log.info("Computing ordinal statistics for standardisation")
        _update_stats(stats, ord_array, image_stack.ordinal_blocks,
                      image_stack.ordinal_missing)
        log.info("Writing ordinal data")
        _standardise_write(ord_array, image_stack.ordinal_blocks,
                           image_stack.ordinal_missing, stats)
        ord_array.attrs.mean = stats.mean
        ord_array.attrs.variance = stats.variance
    else:
        log.info("Writing ordinal data")
        ord_array.attrs.mean = None
        ord_array.attrs.variance = None
        _write(ord_array, image_stack.ordinal_blocks)

    log.info("Closing file")
    h5file.close()
    file_size = os.path.getsize(filename) // (1024 ** 2)
    log.info("Written {}MB file to disk.".format(file_size))


def _to_masked(array: np.ndarray, missing_values: MissingValueList) \
        -> np.ma.MaskedArray:
    """Create a masked array from array plus list of missing."""
    assert len(missing_values) == array.shape[-1]
    mask = np.zeros_like(array, dtype=bool)
    for i, m in enumerate(missing_values):
        if m:
            mask[..., i] = array[..., i] == m
    marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


def _update_stats(stats: _Statistics,
                  array: tables.CArray,
                  blocks: Callable[[], Iterator[np.ndarray]],
                  missing_values: MissingValueList) -> None:
    """Compute the mean and variance of the data."""
    nbands = array.atom.shape[0]
    for b in blocks():
        bs = b.reshape((-1, nbands))
        bm = _to_masked(bs, missing_values)
        stats.update(bm)


def _standardise_write(array: tables.CArray,
                       blocks: Callable[[], Iterator[np.ndarray]],
                       missing_values: MissingValueList,
                       stats: _Statistics) -> None:

    """Write out standardised data."""
    start_idx = 0
    for b in blocks():
        end_idx = start_idx + b.shape[0]
        bm = _to_masked(b, missing_values)
        bm -= stats.mean
        bm /= np.sqrt(stats.variance)
        array[start_idx:end_idx] = bm.data
        start_idx = end_idx

def _categorical_write(array: tables.CArray,
                       blocks: Callable[[], Iterator[np.ndarray]],
                       categories: _Categories):
    """Write without standardising."""
    start_idx = 0
    for b in blocks():
        new_b = categories.update(b)
        end_idx = start_idx + b.shape[0]
        array[start_idx:end_idx] = new_b
        start_idx = end_idx


def _write(array: tables.CArray,
           blocks: Callable[[], Iterator[np.ndarray]]) -> None:
    """Write without standardising or embedding."""
    start_idx = 0
    for b in blocks():
        end_idx = start_idx + b.shape[0]
        array[start_idx:end_idx] = b
        start_idx = end_idx
