"""Input/output routines for geo data types."""

import os.path
import logging
from collections import namedtuple

import rasterio
import numpy as np
import tables
import shapefile
# for mypy type checking
from typing import Callable, Any, List, Tuple, Iterator, Union
from mypy_extensions import NoReturn
from rasterio._io import RasterReader

from landshark.image import pixel_coordinates


log = logging.getLogger(__name__)

# Convenience types
Band = namedtuple("Band", ["image", "index"])
BandCollection = namedtuple("BandCollection", ["ordinal", "categorical"])

# Typechecking aliases
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ShpFieldsType = List[Tuple[str, str, int, int]]


def _match(f: Callable[[Any], Any],
           images: List[RasterReader],
           name: str) -> Any:
    """Return specified property of images if they match."""
    property_list = [f(k) for k in images]
    property_set = set(property_list)
    if len(property_set) != 1:
        _fatal_mismatch(property_list, images, name)
    result = property_set.pop()
    return result


def _fatal_mismatch(property_list: List[Any],
                    images: List[RasterReader],
                    name: str) -> NoReturn:
    """Print a fatal log with helpful table of property mismatch."""
    assert len(property_list) == len(images)

    image_names = [os.path.basename(k.name) for k in images]
    props = zip(image_names, property_list)
    strings = ["{}: {}".format(i, l) for i, l in props]
    table = "\n".join(strings)
    log.error("No match for {}:\n{}".format(name, table))
    raise ValueError("No match for input image property {}".format(name))


def _names(band_list: List[Band]) -> List[str]:
    """Generate a list of band names."""
    band_names = []
    for im, band_idx in band_list:
        basename = os.path.basename(im.name)
        if im.count > 1:
            name = basename + "_{}".format(band_idx)
        else:
            name = basename
        band_names.append(name)
    return band_names


def _missing(bands: List[Band], dtype: np.dtype) -> List[Any]:
    """
    Convert missing data values to a given dtype (rasterio workaround).

    Note that the list may contain 'None' where there are no missing values.
    """
    l = [b.image.nodatavals[b.index - 1] for b in bands]

    def convert(x: Union[None, float]) -> Any:
        return dtype(x) if x is not None else x
    r = [convert(k) for k in l]
    return r


def _bands(images: List[RasterReader]) -> BandCollection:
    """Get list of ordinaal and categorical bands from list of images."""
    # Dont use np.int32 etc here for weird comparison breakage
    categ_types = ["int32", "int64", "uint8", "uint32", "uint64", "uint8"]
    categ_set = {np.dtype(k) for k in categ_types}
    ordin = []
    categ = []
    for im in images:
        for i, d in enumerate(im.dtypes):
            band = Band(image=im, index=(i + 1))   # bands start from 1
            if np.dtype(d) in categ_set:
                categ.append(band)
            else:
                ordin.append(band)

    result = BandCollection(ordinal=ordin, categorical=categ)
    return result


def _block_rows(bands: List[Band]) -> int:
    """Choose a sensible (global) blocksize based on input images' blocks."""
    block_list = []
    for b in bands:
        block = b.image.block_shapes[b.index - 1]
        if not block[0] <= block[1]:
            raise ValueError("No support for column-wise blocks")
        block_list.append(block[0])
    blockrows = int(np.amax(block_list))  # LCM would be more efficient but meh
    return blockrows


def _windows(width: int, height: int,
             block_rows: int) -> List[WindowType]:
    """Create a list of windows to cover the image."""
    assert width > 0
    assert height > 0
    assert block_rows > 0
    n = height // block_rows
    n = n + 1 if (height % block_rows) != 0 else n
    ret = [((i * block_rows, (i + 1) * block_rows), (0, width))
           for i in range(n)]
    return ret


def _block_pixels(width: int, height: int, block_rows: int) -> int:
    """Compute the number of pixels per band in a window."""
    n = width * height * block_rows
    return n


def _read(band_list: List[Band], block_pixels: int, windows: List[WindowType],
          dtype: np.dtype) -> Iterator[np.ndarray]:
    """Create a generator that yields blocks of the image stack."""
    nbands = len(band_list)
    out_array = np.zeros((block_pixels, nbands), dtype=dtype)
    for w in windows:
        for i, b in enumerate(band_list):
            a = b.image.read(b.index, window=w)
            out_array[:, i] = a.astype(dtype).flatten()
        yield out_array


class ImageStack:
    """A stack of registered images with the same res and bbox.

    This class simplifies the handling of multiple geotiff files
    in order to emulate the behaviour of a single many-banded image.

    Parameters
    ----------
    path_list : List[str]
        The paths to image files that will comprise the stack.
    block_rows : Union[None, int]
        Optional integer > 0 that specifies the number of rows read at a time.
        If not provided then a semi-sensible value is computed.

    """

    def __init__(self, path_list: List[str],
                 block_rows: Union[None, int] = None) -> None:
        """Construct an instance of ImageStack."""
        images = [rasterio.open(k, "r") for k in path_list]
        width = _match(lambda x: x.width, images, "width")
        height = _match(lambda x: x.height, images, "height")
        affine = _match(lambda x: x.affine, images, "affine")
        coords_x, coords_y = pixel_coordinates(width, height, affine)
        # crs = _match(lambda x: str(x.crs.data), images, "crs")
        # TODO affine transform
        bands = _bands(images)
        ordinal_bands = bands.ordinal
        categorical_bands = bands.categorical
        ordinal_names = _names(ordinal_bands)
        categorical_names = _names(categorical_bands)
        ordinal_dtype = np.float32
        categorical_dtype = np.int32
        ordinal_missing = _missing(ordinal_bands, dtype=ordinal_dtype)
        categorical_missing = _missing(categorical_bands,
                                       dtype=categorical_dtype)
        if not block_rows:
            log.info("Using block size of {} rows".format(block_rows))
            block_rows = max(_block_rows(ordinal_bands),
                             _block_rows(categorical_bands))
        else:
            log.info("User set block size of {} rows".format(block_rows))
        windows = _windows(width, height, block_rows)
        block_pixels = _block_pixels(width, height, block_rows)
        log.info("Found {} ordinal bands".format(len(ordinal_bands)))
        log.info("Found {} categorical bands".format(len(categorical_bands)))
        log.info("Image resolution is {} x {}".format(width, height))

        self.categorical_dtype = categorical_dtype
        self.ordinal_dtype = ordinal_dtype
        self.width = width
        self.height = height
        self.affine = affine
        self.coordinates_x = coords_x
        self.coordinates_y = coords_y
        self.block_rows = block_rows
        self.windows = windows
        self.block_pixels = block_pixels
        self.ordinal_bands = ordinal_bands
        self.categorical_bands = categorical_bands
        self.ordinal_names = ordinal_names
        self.categorical_names = categorical_names
        self.ordinal_missing = ordinal_missing
        self.categorical_missing = categorical_missing

    def categorical_blocks(self) -> Iterator[np.ndarray]:
        """
        Create an iterator over categorical blocks from the image stack.

        The iterator will pass through the data only once.

        Returns
        -------
        gen : Iterator[np.ndarray]
            An iterator that has categorical (int-valued) image segments
            with block_rows rows of data from the image stack.

        """
        gen = _read(self.categorical_bands, self.block_pixels, self.windows,
                    self.categorical_dtype)
        return gen

    def ordinal_blocks(self) -> Iterator[np.ndarray]:
        """
        Create an iterator over ordinal blocks from the image stack.

        The iterator will pass through the data only once.

        Returns
        -------
        gen : Iterator[np.ndarray]
            An iterator that has ordinal (float-valued) image segments
            with block_rows rows of data from the image stack.

        """
        gen = _read(self.ordinal_bands, self.block_pixels, self.windows,
                    self.ordinal_dtype)
        return gen


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

    n = image_stack.width * image_stack.height
    nbands_cat = len(image_stack.categorical_bands)
    nbands_ord = len(image_stack.ordinal_bands)
    cat_atom = tables.Int32Atom(shape=(nbands_cat,))
    ord_atom = tables.Float32Atom(shape=(nbands_ord,))
    filters = tables.Filters(complevel=1, complib="blosc:lz4")

    log.info("Creating data arrays")
    cat_array = h5file.create_carray(h5file.root, name="categorical_data",
                                     atom=cat_atom, shape=(n,),
                                     filters=filters)
    cat_array.attrs.labels = image_stack.categorical_names
    cat_array.attrs.missing_values = image_stack.categorical_missing
    ord_array = h5file.create_carray(h5file.root, name="ordinal_data",
                                     atom=ord_atom, shape=(n,),
                                     filters=filters)
    ord_array.attrs.labels = image_stack.ordinal_names
    ord_array.attrs.missing_values = image_stack.ordinal_missing

    start_idx = 0
    log.info("Writing categorical data")
    for b in image_stack.categorical_blocks():
        end_idx = start_idx + b.shape[0]
        cat_array[start_idx:end_idx] = b
        start_idx = end_idx

    start_idx = 0
    log.info("Writing ordinal data")
    for b in image_stack.ordinal_blocks():
        end_idx = start_idx + b.shape[0]
        ord_array[start_idx:end_idx] = b
        start_idx = end_idx

    log.info("Closing file")
    h5file.close()
    file_size = os.path.getsize(filename) // (1024 ** 2)
    log.info("Written {}MB file to disk.".format(file_size))


def _shapefile_float_fields(fields: ShpFieldsType) \
        -> Iterator[Tuple[int, str]]:
    """Pull out the float fields from the shapefile field specification.

    Parameters
    ----------
    fields : ShpFieldsType
        The weird list-of-lists that shapefile uses to describe field types

    Returns
    -------
    result : Iterator[Tuple[int, str]]
        An iterator over (<index_number>, <name>) pairs of the float columns.

    """
    shapefields = [f[0] for f in fields[1:]]  # Skip DeletionFlag
    dtype_flags = [(f[1], f[2]) for f in fields[1:]]  # Skip DeletionFlag
    # http://www.dbase.com/Knowledgebase/INT/db7_file_fmt.htm
    # We're only going to support float types for now
    field_indices = [i for i, k in enumerate(dtype_flags) if k[0] == "N"]
    field_names = [shapefields[i] for i in field_indices]
    result = zip(field_indices, field_names)
    return result


class ShapefileTargets:
    """
    Targets for spatial inference backed by a shapefile.

    This class reads a shapefile with point information and provides
    iterators to the coordinates and the (float) records.

    Parameters
    ----------
    filename : str
        The shapefile (.shp)

    """

    def __init__(self, filename: str) -> None:
        """Construct an instance of ShapefileTargets."""
        self._sf = shapefile.Reader(filename)
        self.n = self._sf.numRecords
        self.dtype = np.float32
        float_fields = _shapefile_float_fields(self._sf.fields)
        self._field_indices, self.fields = zip(*float_fields)
        self._field_indices = list(self._field_indices)
        self.fields = list(self.fields)

    def coordinates(self) -> Iterator[np.ndarray]:
        """Create an iterator for the coordinate data.

        This will return a single coordinate from a point, in order
        from the shapefile.

        Returns
        -------
        res : Iterator[np.ndarray]
            An iterator over shape (2,) arrays of x,y coordinates.

        """
        for shape in self._sf.iterShapes():
            res = np.array(shape.__geo_interface__["coordinates"],
                           dtype=np.float64)
            yield res

    def ordinal_data(self) -> Iterator[np.ndarray]:
        """Create an iterator for the ordinal (target) data.

        This will return data at a single point per iteration,
        in order from the shapefile.

        Returns
        -------
        res : Iterator[np.ndarray]
            An iterator over shape (k,) of k records for each point.

        """
        for rec in self._sf.iterRecords():
            res = np.array([rec[i] for i in self._field_indices],
                           dtype=self.dtype)
            yield res


def write_targetfile(sf: ShapefileTargets, filename: str) -> None:
    """
    Write out a representation of target data to an HDF5 from a shapefile.

    Parameters
    ----------
    sf : ShapefileTargets
        The shapefile object to output.
    filename : str
        The output filename of the HDF5 file.

    """
    title = "Landshark Targets"
    log.info("Creating HDF5 target file")
    h5file = tables.open_file(filename, mode="w", title=title)

    n = sf.n
    ncols_ord = len(sf.fields)
    ord_atom = tables.Float32Atom()
    filters = tables.Filters(complevel=1, complib="blosc:lz4")

    log.info("Creating data arrays")
    target_array = h5file.create_carray(h5file.root, name="targets",
                                        atom=ord_atom, shape=(n, ncols_ord),
                                        filters=filters)
    target_array.attrs.labels = sf.fields

    coord_array = h5file.create_carray(h5file.root, name="coordinates",
                                       atom=ord_atom, shape=(n, 2),
                                       filters=filters)
    coord_array.attrs.labels = ["x", "y"]

    log.info("Writing target data")
    for i, r in enumerate(sf.ordinal_data()):
        target_array[i] = r

    log.info("Writing coordinate data")
    for i, c in enumerate(sf.coordinates()):
        coord_array[i] = c

    log.info("Closing file")
    h5file.close()
    file_size = os.path.getsize(filename) // (1024 ** 2)
    log.info("Written {}MB file to disk.".format(file_size))
