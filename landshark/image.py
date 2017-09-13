"""Image operations that move between world and image coordinates."""
import logging
from collections import namedtuple

import numpy as np
from affine import Affine
# mypy type checking
from typing import Tuple

log = logging.getLogger(__name__)


BoundingBox = namedtuple("BoundingBox", ["x0", "xn", "y0", "yn"])


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
    pix_x = np.arange(width + 1, dtype=np.float64)
    pix_y = np.arange(height + 1, dtype=np.float64)
    coords_x = (pix_x * pixel_width) + origin_x
    coords_y = (pix_y * pixel_height) + origin_y

    return coords_x, coords_y


def bounds(x_pixel_coords: np.ndarray,
           y_pixel_coords: np.ndarray) -> BoundingBox:
    """
    Get the bounding box of the image.

    Returns the corners at 0,0 and (width, height).

    Parameters
    ----------
    x_pixel_coords : np.ndarray
        Array of pixel coordinates in world space. each edge must be
    the minimum mag side, and it must extend one pixel beyond
    y_pixel_coords : np.ndarray
        Array of pixel coordinates in y. See x_pixel_coords.

    Returns
    -------
    bbox : BoundingBox
        Struct giving the 4 coordinates of the bounding box. They are
        labelled x0 and y0 for the corner of the 0,0 pixel, and xn yn for
        the (outer) corner of the last pixel.

    """
    bbox = BoundingBox(x0=x_pixel_coords[0], xn=x_pixel_coords[-1],
                       y0=y_pixel_coords[0], yn=y_pixel_coords[-1])
    return bbox


def in_bounds(coords_x: np.ndarray, coords_y: np.ndarray,
              bbox: BoundingBox) -> np.ndarray:
    """Do Stuff."""
    minx = min(bbox.x0, bbox.xn)
    maxx = max(bbox.x0, bbox.xn)
    miny = min(bbox.y0, bbox.yn)
    maxy = max(bbox.y0, bbox.yn)

    in_x = np.logical_and(coords_x >= minx, coords_x <= maxx)
    in_y = np.logical_and(coords_y >= miny, coords_y <= maxy)
    in_bbox = np.logical_and(in_x, in_y)
    return in_bbox


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
    assert pixel_coordinate_array.dtype == np.float64
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
