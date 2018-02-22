"""Image operations that move between world and image coordinates."""
import logging
from itertools import product

import numpy as np
from rasterio.transform import from_bounds
from affine import Affine
from typing import Tuple, Iterable, Dict, List, Optional, Any

from landshark import iteration
from landshark.basetypes import FixedSlice, CoordinateType

log = logging.getLogger(__name__)


class BoundingBox:
    """
    The bounding box of an image.

    Parameters
    ----------
    x_pixel_coords : np.ndarray
        Array of pixel coordinates in world space. each edge must be
    the minimum mag side, and it must extend one pixel beyond
    y_pixel_coords : np.ndarray
        Array of pixel coordinates in y. See x_pixel_coords.

    """

    def __init__(self, x_pixel_coords: np.ndarray,
                 y_pixel_coords: np.ndarray) -> None:
        """Construct the bounding box."""
        assert x_pixel_coords.ndim == 1
        assert y_pixel_coords.ndim == 1
        x0 = x_pixel_coords[0]
        xn = x_pixel_coords[-1]
        y0 = y_pixel_coords[0]
        yn = y_pixel_coords[-1]
        self.xmin = min(x0, xn)
        self.xmax = max(x0, xn)
        self.ymin = min(y0, yn)
        self.ymax = max(y0, yn)

    def __repr__(self) -> str:
        """Print a string representation of object."""
        rep = "<bbox ([{:.2f},{:.2f}], [{:.2f},{:.2f}])>".format(
            self.xmin, self.xmax, self.ymin, self.ymax)
        return rep

    def contains(self, coords: np.ndarray) -> np.ndarray:
        """
        Check membership of coordinates in the bbox.

        This assumes the bbox is closed: points on the boundary are in the box.

        Parameters
        ----------
        coords : np.ndarray
            A (k, 2) array of k 2D points to test for membership

        Returns
        -------
        in_bbox : np.array
            A (k,) shape boolean array specifying whether each point
            is inside the bbox

        """
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        coords_x = coords[:, 0]
        coords_y = coords[:, 1]
        in_x = np.logical_and(coords_x >= self.xmin, coords_x <= self.xmax)
        in_y = np.logical_and(coords_y >= self.ymin, coords_y <= self.ymax)
        in_bbox = np.logical_and(in_x, in_y)
        return in_bbox


class ImageSpec:
    """
    Struct encapsulating the geographical information about an image.

    Parameters
    ----------
    x_coordinates : np.ndarray
        The x-coordinates of every pixel edge starting from the 0th pixel.
        If there are k pixels then there are k + 1 edges.
    y_coordinates : np.ndarray
        The y-coordinates of every pixel edge. See x_coordinates.

    """

    def __init__(self, x_coordinates: np.ndarray,
                 y_coordinates: np.ndarray, crs: Dict[str, str]) -> None:
        """Construct the ImageSpec object."""
        assert x_coordinates.ndim == 1
        assert y_coordinates.ndim == 1
        self.width = x_coordinates.shape[0] - 1
        self.height = y_coordinates.shape[0] - 1
        assert self.width > 0
        assert self.height > 0
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.bbox = BoundingBox(x_coordinates, y_coordinates)
        self.crs = crs

        # affine transformation
        self.affine = from_bounds(west=self.bbox.xmin, east=self.bbox.xmax,
                                  north=self.bbox.ymax, south=self.bbox.ymin,
                                  width=self.width, height=self.height)

    def __repr__(self) -> str:
        """Create a string representation of the object."""
        rep = "<{}x{} image with bbox {} in {}>".format(
            self.height, self.width, self.bbox, self.crs)
        return rep


def pixel_coordinates(width: int,
                      height: int,
                      affine: Affine) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the pixel to coordinate map.

    Note affine follows the standard conventenion that origin is
    the UPPER left. The coordinates are for all edges, so for k pixels
    there are k + 1 coordinates. The types here are all 64bit.

    Parameters
    ----------
    width : int
        The width in pixels of the image.
    height : int
        The height in pixels of the image.
    affine : Affine
        The affine transformation representing the change to world coords.

    Returns
    -------
    coords_x : np.ndarray
        The x-coordinates in world space of the pixel edges.
    coords_y : np.ndarray
        The y-coordinates in wourld space of the pixel edges

    """
    assert affine.is_rectilinear

    pixel_width = affine[0]
    pixel_height = affine[4]
    origin_x = affine[2]
    origin_y = affine[5]

    # construct the canonical pixel<->position map: +1 for outer corner
    pix_x = np.arange(width + 1, dtype=CoordinateType)
    pix_y = np.arange(height + 1, dtype=CoordinateType)
    coords_x = (pix_x * pixel_width) + origin_x
    coords_y = (pix_y * pixel_height) + origin_y

    return coords_x, coords_y


def image_to_world(indices: np.ndarray,
                   pixel_coordinate_array: np.ndarray) -> np.ndarray:
    """
    Map image coordinates (pixel indices) to world coordinates.

    This function uses the canonical pixel_coordinate array of the
    smallest magnitude edges (ie left-most for +ve pixel size, right-most
    for -ve pixel size) to compue the world coordinates of pixel
    indices by a lookup table. Requires pixels to actually be in the image.

    Parameters
    ----------
    indices: np.ndarray
        a 1D array of the indices to look up. Must actually be in the image.
    pixel_coordinate_array : np.ndarray
        a 1-d numpy array of pixel edge coordinates in world space. Each edge
        must be the minimum-magnitude side. The array is assumed to go
        1 past the edge of the image.

    Returns
    -------
    a 1D array of coordinates giving the smallest-magnitude edge
    of each pixel in indices.

    """
    assert indices.ndim == 1
    assert indices.dtype == np.int64
    assert pixel_coordinate_array.ndim == 1
    assert pixel_coordinate_array.dtype == CoordinateType
    assert np.all(indices >= 0)
    assert np.all(indices < (pixel_coordinate_array.shape[0] - 1))

    result = pixel_coordinate_array[indices]
    return result


def world_to_image(points: np.ndarray,
                   pixel_coordinate_array: np.ndarray) -> np.ndarray:
    """
    Map world coordinates to pixel indices.

    This function uses a canonical pixel_coordinate_array mapping and performs
    a lookup to find where the points sit in that array. This means that it
    should provide EXACT pixel -> location -> pixel transforms with zero
    aliasing.

    Parameters
    ----------
    points : np.ndarry
        a 1D array of points to loop up. Must actually be in the image.
    pixel_coordinate_array : np.ndarray
        a 1-d numpy array of pixel edge coordinates in world space. Each edge
        must be the minimum-magnitude side. The array is assumed to go
        1 past the edge of the image.

    Returns
    -------
    A 1D array of ints corresponding to the pixel indices in the image for
    each world point.

    """
    reverse = pixel_coordinate_array[1] < pixel_coordinate_array[0]
    if reverse:
        rev_idx = np.searchsorted(pixel_coordinate_array[::-1], points,
                                  side="left") + 1
        idx = len(pixel_coordinate_array) - rev_idx
    else:
        idx = np.searchsorted(pixel_coordinate_array, points,
                              side="right") - 1

    # We want the *closed* interval, which means moving
    # points on the end back by 1
    on_end = points == pixel_coordinate_array[-1]
    idx[on_end] -= 1

    res = pixel_coordinate_array.shape[0] - 1
    if (not all(np.logical_and(idx >= 0, idx < res))):
        raise ValueError("Queried location is not in the image")
    return idx


def strip_image_spec(strip: int, nstrips: int,
                     image_spec: ImageSpec) -> ImageSpec:
    """
    Create an imagespec for an indexed strip of a larger image.

    Parameters
    ----------
    strip : int
        The index of the strip of the larger image. 1 <= strip <= nstrips.

    nstrips : int
        The total number of strips between which to divide the image.
        Must be greater than 0.

    image_spec : ImageSpec
        The imagespec of the full-sized image that is being divided.

    Returns
    -------
    new_spec : ImageSpec
        The imagespec of the strip.

    """
    assert nstrips > 0
    assert strip >= 1 and strip <= nstrips
    # strips are indexed from one
    strip_slice = _strip_slices(image_spec.height, nstrips)[strip - 1]
    # coordinates are of all pixel edges so need to go one past the end
    x_coords = image_spec.x_coordinates
    y_coords = image_spec.y_coordinates[
        strip_slice.start: strip_slice.stop + 1]
    crs = image_spec.crs
    new_spec = ImageSpec(x_coords, y_coords, crs)
    return new_spec


def indices_strip(image_spec: ImageSpec, strip: int, nstrips: int,
                  batchsize: int) \
        -> Tuple[Iterable[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Create an iterator over each row of a strip.

    Parameters
    ----------
    image_spec : ImageSpec
        The imagespec of the full-sized image.
    strip : int
        The index of the strip. 1 <= strip <= nstrips.
    nstrips : int
        The total number of strips into which to divide the image.
    batchsize : int
        The (maximum) number of rows in each batch from the iterator.

    Returns
    -------
    it : Iterator[np.ndarray, np.ndarray]
        The indices of each batch as x, y arrays.
    n_total : int
        The total number of elements that the iterator passes over.

    """
    assert nstrips > 0
    assert strip >= 1 and strip <= nstrips
    assert batchsize > 0
    slices = _strip_slices(image_spec.height, nstrips)
    s = slices[strip - 1]   # indexed from one
    n_total = (s.stop - s.start) * image_spec.width
    it = _indices_query(image_spec.width, image_spec.height, batchsize,
                        row_slice=s)
    return it, n_total


def _strip_slices(total_size: int, nstrips: int) -> List[FixedSlice]:
    """Compute the slices corresponding to every strip along a dimension."""
    assert nstrips > 0
    assert total_size >= nstrips
    strip_size_small, nbig = divmod(total_size, nstrips)
    strip_size_big = strip_size_small + 1
    strip_sizes = [strip_size_big] * nbig + \
        [strip_size_small] * (nstrips - nbig)
    indices = np.cumsum([0] + strip_sizes)
    slices = [FixedSlice(i, j) for i, j in zip(indices[0:-1], indices[1:])]
    return slices


def _array_pair_it(x: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the x and y coordinate arrays from the batch iterator."""
    a = np.array(x).T[::-1]
    r = (a[0], a[1])
    return r


def _indices_query(
    image_width: int,
    image_height: int,
    batchsize: int,
    column_slice: Optional[FixedSlice]=None,
    row_slice: Optional[FixedSlice]=None
        ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Create a generator of batches of coordinates from an image."""
    column_slice = column_slice if column_slice else FixedSlice(0, image_width)
    row_slice = row_slice if row_slice else FixedSlice(0, image_height)

    height_ind = range(image_height)[row_slice.start: row_slice.stop]
    width_ind = range(image_width)[column_slice.start: column_slice.stop]

    coords_it = product(height_ind, width_ind)
    total_size = len(height_ind) * len(width_ind)
    batch_it = iteration.batch(coords_it, batchsize, total_size)
    array_it = map(_array_pair_it, batch_it)
    return array_it
