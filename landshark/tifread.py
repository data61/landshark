"""Importing routines for tif data."""

from types import TracebackType
import os.path
import logging
from contextlib import ExitStack

import rasterio
from rasterio.io import DatasetReader
from affine import Affine
import numpy as np
from typing import Callable, Any, List, Tuple, Union, Optional, NamedTuple
from mypy_extensions import NoReturn

from landshark.image import pixel_coordinates, ImageSpec
from landshark.basetypes import ArraySource, OrdinalArraySource, \
    CategoricalArraySource, FeatureType


log = logging.getLogger(__name__)

# Typechecking aliases
ShpFieldsType = List[Tuple[str, str, int, int]]
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]


# Convenience types
class Band(NamedTuple):
    image: DatasetReader
    idx: int


def shared_image_spec(path_list: List[str],
                      ignore_crs: bool=False) -> ImageSpec:
    """Get the (hopefully matching) image spec from a list of images"""
    with ExitStack() as stack:
        all_images = [stack.enter_context(rasterio.open(k, "r"))
                      for k in path_list]
        width = _match(lambda x: x.width, all_images, "width")
        height = _match(lambda x: x.height, all_images, "height")
        affine = _match_transforms([x.transform for x in all_images],
                                   all_images)
        coords_x, coords_y = pixel_coordinates(width, height, affine)

    crs = _match(lambda x: x.crs.data if x.crs else None,
                 all_images, "crs", anyof=ignore_crs)
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
    _type_name = ""

    def __init__(self, image_spec: ImageSpec, path_list: List[str],
                 missing: int=-2147483648) -> None:
        """Construct an instance of ImageStack."""
        self._path_list = path_list
        with ExitStack() as stack:
            all_images = [stack.enter_context(rasterio.open(k, "r"))
                          for k in path_list]
            bands = _bands(all_images)
            nbands = len(bands)
            self._shape = (image_spec.height,
                           image_spec.width, nbands)
            self._missing = _missing(missing, bands, self.dtype)
            self._columns = _names(bands)
            self._native = _block_rows(bands)

        log.info("Found {} {} bands".format(nbands, self._type_name))
        log.info("Largest tif block size is {} rows".format(self._native))

    def __enter__(self) -> None:
        self._images = [rasterio.open(k, "r") for k in self._path_list]
        self._bands = _bands(self._images)
        super().__enter__()

    def __exit__(self, ex_type: type, ex_val: Exception,
                 ex_tb: TracebackType) -> None:
        for i in self._images:
            i.close()
        del(self._images)
        del(self._bands)
        super().__exit__(ex_type, ex_val, ex_tb)
        pass

    def _arrayslice(self, start_row: int, end_row: int) -> np.ndarray:
        """Create a generator that yields blocks of the image stack."""
        assert start_row < end_row
        w = ((start_row, end_row), (0, self._shape[1]))
        shape = (end_row - start_row, self._shape[1], self.shape[-1])
        out_array = np.empty(shape, dtype=self._dtype)

        start_band = 0
        for im in self._images:
            stop_band = start_band + im.count
            marray = im.read(window=w, masked=True).astype(self._dtype)
            if self._missing is not None:
                if np.sum(marray.data == self._missing) > 0:
                    raise ValueError("Mask value detected in dataset")
                marray.data[marray.mask] = self._missing
            data = marray.data
            data = np.moveaxis(data, 0, -1)
            out_array[..., start_band:stop_band] = data
            start_band = stop_band

        return out_array


class OrdinalStackSource(_ImageStackSource, OrdinalArraySource):
    _type_name = "ordinal"


class CategoricalStackSource(_ImageStackSource, CategoricalArraySource):
    _type_name = "categorical"


def _match(f: Callable[[Any], Any],
           images: List[DatasetReader],
           name: str,
           anyof: bool=False) -> Any:
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


def _match_transforms(transforms: List[Affine],
                      images: List[DatasetReader]) -> Affine:
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


def _missing(value: FeatureType, bands: List[Band],
             dtype: np.dtype) -> Optional[FeatureType]:
    """
    Convert missing data values to a given dtype (rasterio workaround).

    Note that the list may contain 'None' where there are no missing values.
    """
    r_set = set([b.image.nodatavals[b.idx - 1] for b in bands])
    result = None if r_set == {None} else dtype(value)
    return result


def _bands(images: List[DatasetReader]) -> List[Band]:
    """Get bands from list of images."""
    bandlist = []
    for im in images:
        for i, _ in enumerate(im.dtypes):
            band = Band(image=im, idx=(i + 1))   # bands start from 1
            bandlist.append(band)
    return bandlist


def _block_rows(bands: List[Band]) -> int:
    """Choose a sensible (global) blocksize based on input images' blocks."""
    block_list = []
    for b in bands:
        block = b.image.block_shapes[b.idx - 1]
        if not block[0] <= block[1]:
            raise ValueError("No support for column-wise blocks")
        block_list.append(block[0])
    blockrows = int(np.amax(block_list))  # LCM would be more efficient but meh
    return blockrows
