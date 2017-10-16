"""Read features and targets from HDF5 files."""

import numpy as np
import tables
from typing import Iterable, Tuple, Union, List, Any

from landshark import image
from landshark.rowcache import RowCache


class ImageFeatures:
    """Reader for getting categorical and ordinal features from an HDF5 file.

    Parameters
    ----------
    filename : str
        The path the the HDF5 file containing the image data.
    cache_blocksize : int
        The blocksize (in rows) to use for caching the reads of the HDF5.
    cache_nblocks : int
        The number of blocks to hold in a cache at any time.

    """

    def __init__(self, filename: str, cache_blocksize: int,
                 cache_nblocks: int) -> None:
        """Initialise an ImageFeatures object."""
        self._hfile = tables.open_file(filename)
        x_coordinates = self._hfile.root.x_coordinates.read()
        y_coordinates = self._hfile.root.y_coordinates.read()
        height = self._hfile.root._v_attrs.height
        width = self._hfile.root._v_attrs.width
        assert len(x_coordinates) == width + 1
        assert len(y_coordinates) == height + 1
        spec = image.ImageSpec(x_coordinates, y_coordinates)
        self.image_spec = spec
        self.ord = Features(
            self._hfile.root.ordinal_data,
            cache_blocksize,
            cache_nblocks
            )
        self.cat = Features(
            self._hfile.root.categorical_data,
            cache_blocksize,
            cache_nblocks
            )

    def pixel_indices(self, batchsize: int) \
            -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Create a generator of batches of coordinates from an image.

        This will iterate through ALL of the pixel coordinates an HDF5 file,
        so is useful for querying/prediction.

        Parameters
        ----------
        batchsize : int
            the number of coorinates to yield at once.

        Yields
        ------
        im_coords_x : ndarray
            the x coordinates (width) of the image in pixels indices, of shape
            (batchsize,).
        im_coords_y : ndarray
            the y coordinates (height) of the image in pixels indices, of shape
            (batchsize,).

        """
        pixel_it = image.coords_query(self.image_spec.width,
                                      self.image_spec.height, batchsize)
        return pixel_it


class Features:
    """Reader for getting image features from an HDF5 array.

    Parameters
    ----------
    carray : tables.carray.CArray
        Then HDF5 array where the data is stored.
    missing_values : list
        The missing data values per feature index in ``carray``.
    cache_blocksize : int
        The blocksize (in rows) to use for caching the reads of the HDF5.
    cache_nblocks : int
        The number of blocks to hold in a cache at any time.

    """

    def __init__(
            self,
            carray: tables.carray.CArray,
            cache_blocksize: int,
            cache_nblocks: int
            ) -> None:
        """Initialise a Features object."""
        self._carray = carray
        self._missing_values = carray.attrs.missing_values
        self._cache = RowCache(carray, cache_blocksize, cache_nblocks)

    @property
    def nfeatures(self) -> Any:
        """Get the number of features in the HDF5 file."""
        return self._carray.atom.shape[0]

    @property
    def dtype(self) -> Any:
        """Get the type of features in the HDF5 file."""
        return self._carray.atom.dtype.base

    @property
    def missing_values(self) -> List[Union[None, float, int]]:
        """Get the list of missing values for each feature."""
        return self._missing_values

    def __call__(self, y: int, x_slice: slice) -> np.ndarray:
        """Read values from the HDF5 file.

        Parameters
        ----------
        y : int
            The row (y) index of the slice.
        x_slice: slice
            The slice in x over the row specificed by idx.

        Returns
        -------
        d : np.ndarray
            The contiguous block of data from that location in the image.

        """
        return self._cache(y, x_slice)


class Targets:
    """Reader for getting targets and coordinates from an HDF5 file.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file contains the target information.
    label : str
        The name of the target to read.

    """

    def __init__(self, filename: str, label: str) -> None:
        """Initialise a Targets object."""
        self._hfile = tables.open_file(filename)
        labels = self._hfile.root.targets.attrs.labels
        label_index = labels.index(label)
        # TODO don't read the whole file into memory
        self._data = self._hfile.root.targets[:, label_index]
        self.coordinates = self._hfile.root.coordinates

    def training(self, image_spec: image.ImageSpec, batchsize: int) \
            -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate batches of target coordinated and values.

        Parameters
        ----------
        image_spec : ImageSpec
            the image specifactions of the features file.
        batchsize : int
            the number of targets to generate per batch

        Yields
        ------
        px : np.ndarray
            array of image/feature "x" indices corresponding to the targets.
        py : np.ndarray
            array of image/feature "y" indices corresponding to the targets.
        target : np.ndarray
            the target values for each index.

        """
        pixel_it = image.coords_training(self.coordinates,
                                         image_spec.x_coordinates,
                                         image_spec.y_coordinates, batchsize)
        start = 0
        for px, py in pixel_it:
            n = px.shape[0]
            stop = start + n
            s = slice(start, stop)
            target = self._data[s]
            start = stop
            yield px, py, target
