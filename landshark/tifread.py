"""Importing routines for tif data."""

import os.path
import logging
from collections import namedtuple

import rasterio
from rasterio.io import DatasetReader
import numpy as np
from typing import Callable, Any, List, Tuple, Union
from mypy_extensions import NoReturn

from landshark.image import pixel_coordinates, ImageSpec
from landshark.basetypes import ArraySource, OrdinalArraySource, \
    CategoricalArraySource


log = logging.getLogger(__name__)

# Typechecking aliases
ShpFieldsType = List[Tuple[str, str, int, int]]
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]

# Convenience types
Band = namedtuple("Band", ["image", "index"])


def shared_image_spec(path_list: List[str]) -> ImageSpec:
    """Get the (hopefully matching) image spec from a list of images"""
    all_images = [rasterio.open(k, "r") for k in path_list]
    width = _match(lambda x: x.width, all_images, "width")
    height = _match(lambda x: x.height, all_images, "height")
    affine = _match_transforms([x.transform for x in all_images], all_images)
    coords_x, coords_y = pixel_coordinates(width, height, affine)
    crs = _match(lambda x: x.crs.data, all_images, "crs", anyof=True)
    imspec = ImageSpec(coords_x, coords_y, crs)
    return imspec


class _ImageStackSource(ArraySource):
    """A stack of registered images with the same res and bbox.

    This class simplifies the handling of multiple geotiff files
    in order to emulate the behaviour of a single many-banded image.

    Parameters
    ----------
    path_list : List[str]
        The list of images to stack.
    block_rows : Union[None, int]
        Optional integer > 0 that specifies the number of rows read at a time.
        If not provided then a semi-sensible value is computed.

    """
    def __init__(self, spec: ImageSpec, path_list: List[str]) -> None:
        """Construct an instance of ImageStack."""
        all_images = [rasterio.open(k, "r") for k in path_list]
        self._bands = _bands(all_images)
        nbands = len(self._bands)
        self._shape = (spec.height, spec.width, nbands)
        self._missing = _missing(self._bands, dtype=self.dtype)
        self._columns = _names(self._bands)
        self._native = _block_rows(self._bands)

        log.info("Found {} {} bands".format(nbands, self.dtype))
        log.info("Using tif block size of {} rows".format(self._native))

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        array = _read_slice(self._bands, start, end, self._shape[1],
                            self.dtype)
        return array


class OrdinalStackArraySource(_ImageStackSource, OrdinalArraySource):
    pass

class CategoricalStackArraySource(_ImageStackSource, CategoricalArraySource):
    pass

# class OrdinalStackSource(OrdinalDataSource):
#     def __init__(self, spec: ImageSpec, filenames: str):
#         source = OrdinalStackArraySource(spec, filenames)
#         super().__init__(source)

# class CategoricalStackSource(CategoricalDataSource):
#     def __init__(self, spec: ImageSpec, filenames: str):
#         source = CategoricalStackArraySource(spec, filenames)
#         super().__init__(source)


def _match(f: Callable[[Any], Any],
           images: List[DatasetReader],
           name: str,
           anyof=False) -> Any:
    """Return specified property of images if they match."""
    property_list = [f(k) for k in images]
    if len(images) == 1:
        allequal = True
    else:
        allequal = all(p == property_list[0] for p in property_list[1:])

    if not (allequal or anyof):
        _fatal_mismatch(property_list, images, name)
    result = property_list[0]
    return result


def _match_transforms(transforms, images):
    t0 = transforms[0]
    for t in transforms[1:]:
        if not t0.almost_equals(t):
            _fatal_mismatch(transforms, images, "transforms")
    return t0


def _fatal_mismatch(property_list: List[Any],
                    images: List[DatasetReader],
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
    lst = [b.image.nodatavals[b.index - 1] for b in bands]

    def convert(x: Union[None, float]) -> Any:
        return dtype(x) if x is not None else x
    r = [convert(k) for k in lst]
    return r


def _bands(images: List[DatasetReader]) -> List[Band]:
    """Get bands from list of images."""
    bandlist = []
    for im in images:
        for i, _ in enumerate(im.dtypes):
            band = Band(image=im, index=(i + 1))   # bands start from 1
            bandlist.append(band)
    return bandlist


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


def _read_slice(band_list: List[Band], start_row: int, end_row: int,
                width: int, dtype: np.dtype) -> np.ndarray:
    """Create a generator that yields blocks of the image stack."""
    assert width > 0
    assert start_row < end_row
    w = ((start_row, end_row), (0, width))
    shape = (end_row - start_row, width, len(band_list))
    out_array = np.empty(shape, dtype=dtype)
    for i, b in enumerate(band_list):
        a = b.image.read(b.index, window=w)
        out_array[:, :, i] = a.astype(dtype)
    return out_array
