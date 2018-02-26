"""Importing routines for tif data."""

import os.path
import logging

import rasterio
from rasterio.io import DatasetReader
import numpy as np
from typing import Callable, Any, List, Tuple, Union
from mypy_extensions import NoReturn

from landshark.image import pixel_coordinates, ImageSpec
from landshark.basetypes import ArraySource, OrdinalArraySource, \
    CategoricalArraySource, OrdinalType, CategoricalType

log = logging.getLogger(__name__)

# Typechecking aliases
ShpFieldsType = List[Tuple[str, str, int, int]]
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]


def shared_image_spec(path_list: List[str]) -> ImageSpec:
    """Get the (hopefully matching) image spec from a list of images"""
    all_images = [rasterio.open(k, "r") for k in path_list]
    width = _match(lambda x: x.width, all_images, "width")
    height = _match(lambda x: x.height, all_images, "height")
    affine = _match_transforms([x.transform for x in all_images], all_images)
    coords_x, coords_y = pixel_coordinates(width, height, affine)
    crs = _match(lambda x: str(x.crs.data), all_images, "crs", anyof=True)
    imspec = ImageSpec(coords_x, coords_y, crs)
    return imspec


class _ImageSource(ArraySource):
    def __init__(self, spec: ImageSpec, path: str,
                 missing: int=-2147483648) -> None:
        self._path = path
        self._missing = self._dtype(missing)
        with rasterio.open(self._path, "r") as rfile:
            nbands = rfile.count
            self._shape = (rfile.height, rfile.width, nbands)
            self._orig_missing = _missing(rfile, dtype=self.dtype)
            if len(set(self._orig_missing)) == 1 and \
                    self._orig_missing[0] is None:
                self._missing = None
            self._columns = _names(self._path, nbands)
            self._native = _block_rows(rfile)
            assert len(self._columns) == nbands
        self.name = os.path.basename(path)
        log.info("{} has {} {} bands, {} row blocks, missing values {}".format(
            self.name, nbands, self._type_str,
            self._native, self._orig_missing))

    def __enter__(self):
        self._rfile = rasterio.open(self._path, "r")
        super().__enter__()

    def __exit__(self, *args):
        self._rfile.close()
        del(self._rfile)
        super().__exit__(*args)

    def _arrayslice(self, start: int, end: int) -> np.ndarray:
        w = ((start, end), (0, self.shape[1]))
        marray = self._rfile.read(window=w, masked=True).astype(self.dtype)
        if self._missing is not None:
            if np.sum(marray.data == self._missing) > 0:
                raise ValueError("Mask value detected in dataset")
            marray.data[marray.mask] = self._missing
        data = marray.data
        data = np.moveaxis(data, 0, -1)
        return data


class OrdinalImageSource(_ImageSource, OrdinalArraySource):
    _type_str = "ordinal"

class CategoricalImageSource(_ImageSource, CategoricalArraySource):
    _type_str = "categorical"

def _match(f: Callable[[Any], Any],
           images: List[DatasetReader],
           name: str,
           anyof=False) -> Any:
    """Return specified property of images if they match."""
    property_list = []
    for k in images:
        try:
            property_list.append(f(k))
        except:
            continue
    property_set = set(property_list)
    if len(property_set) != 1 and not anyof:
        _fatal_mismatch(property_list, images, name)
    result = property_set.pop()
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


def _names(path, nbands) -> List[str]:
    """Generate a list of band names."""
    basename = os.path.basename(path)
    if nbands == 1:
        band_names = [basename]
    else:
        band_names = [basename + "_{}".format(i + 1) for i in range(nbands)]
    return band_names


def _missing(image, dtype: np.dtype) -> List[Any]:
    """
    Convert missing data values to a given dtype (rasterio workaround).

    Note that the list may contain 'None' where there are no missing values.
    """
    lst = image.nodatavals

    def convert(x: Union[None, float]) -> Any:
        return dtype(x) if x is not None else x
    r = [convert(k) for k in lst]
    return r


def _block_rows(image) -> int:
    """Choose a sensible (global) blocksize based on input images' blocks."""
    block_list = []
    for b in range(image.count):
        block = image.block_shapes[b]
        if not block[0] <= block[1]:
            raise ValueError("No support for column-wise blocks")
        block_list.append(block[0])
    blockrows = int(np.amax(block_list))  # LCM would be more efficient but meh
    return blockrows
