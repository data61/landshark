"""Importing routines for tif data."""

import os.path
import logging

import numpy as np
import tables

from landshark.importers.tifread import ImageStack

log = logging.getLogger(__name__)


class _Statistics:
    """Class that computes online mean and variance."""

    def __init__(self, n_features):
        """Initialise the counters."""
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._n = np.zeros(n_features, dtype=int)

    def update(self, array: np.ma.MaskedArray) -> None:
        """Update calclulations with new data."""
        assert array.ndim == 2
        assert array.shape[0] > 1

        new_n = np.ma.count(array, axis=0)
        new_mean = np.ma.mean(array, axis=0)
        new_m2 = np.ma.var(array, axis=0, ddof=0) * new_n

        delta = new_mean - self._mean
        delta_mean = delta * (new_n / (new_n + self._n))

        self._mean += delta_mean
        self._m2 += new_m2 + (delta * self._n * delta_mean)
        self._n += new_n

    @property
    def mean(self) -> np.ndarray:
        """Get the current estimate of the mean."""
        assert np.all(self._n > 1)
        return self._mean

    @property
    def var(self) -> np.ndarray:
        """Get the current estimate of the variance."""
        assert np.all(self._n > 1)
        var = self._m2 / self._n
        return var


def _to_masked(array, missing_values):
    """Create a masked array from array plus list of missing."""
    assert len(missing_values) == array.shape[-1]

    mask = np.zeros_like(array, dtype=bool)
    for i, m in enumerate(missing_values):
        if m:
            mask[..., i] = array[..., i] == m
    marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


def write_datafile(image_stack: ImageStack, filename: str) -> None:
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

    """
    title = "Landshark Image Stack"
    log.info("Creating HDF5 output file")
    h5file = tables.open_file(filename, mode="w", title=title)

    # write the attributes to root
    log.info("Writing global attributes")
    attributes = h5file.root._v_attrs
    attributes.height = image_stack.height
    attributes.width = image_stack.width
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
    cat_array.attrs.missing_values = image_stack.categorical_missing
    ord_array = h5file.create_carray(h5file.root, name="ordinal_data",
                                     atom=ord_atom, shape=im_shape,
                                     filters=filters)
    ord_array.attrs.labels = image_stack.ordinal_names
    ord_array.attrs.missing_values = image_stack.ordinal_missing

    log.info("Categorical HDF5 block shape: {}".format(cat_array.chunkshape))
    log.info("Ordinal HDF5 block shape: {}".format(ord_array.chunkshape))

    start_idx = 0
    log.info("Writing categorical data")
    for b in image_stack.categorical_blocks():
        end_idx = start_idx + b.shape[0]
        cat_array[start_idx:end_idx] = b
        start_idx = end_idx

    ord_stats = _Statistics(nbands_ord)

    log.info("Computing statistics")
    cat_array.attrs.mean = None
    cat_array.attrs.variance = None
    start_idx = 0
    for b in image_stack.ordinal_blocks():
        bm = _to_masked(b.reshape((-1, nbands_ord)),
                        image_stack.ordinal_missing)
        ord_stats.update(bm)
        end_idx = start_idx + b.shape[0]
        start_idx = end_idx

    ord_array.attrs.mean = ord_stats.mean
    ord_array.attrs.variance = ord_stats.var

    start_idx = 0
    log.info("Writing ordinal data")
    for b in image_stack.ordinal_blocks():
        bm = _to_masked(b, image_stack.ordinal_missing)
        bm -= ord_stats.mean
        bm /= np.sqrt(ord_stats.var)
        end_idx = start_idx + b.shape[0]
        ord_array[start_idx:end_idx] = bm.data
        start_idx = end_idx

    log.info("Closing file")
    h5file.close()
    file_size = os.path.getsize(filename) // (1024 ** 2)
    log.info("Written {}MB file to disk.".format(file_size))
