"""Image operations that move between world and image coordinates."""
import logging
from itertools import product, islice

import numpy as np
from affine import Affine
# mypy type checking
from typing import Tuple, Iterable

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
                 y_coordinates: np.ndarray) -> None:
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


def coords_query(
    image_width: int,
    image_height: int,
    batchsize: int
        ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Create a generator of batches of coordinates from an image.

    This will iterate through ALL of the pixel coordinates in an image, so is
    useful for querying/prediction.

    Parameters
    ----------
    image_width : int
        the number of pixels wide the image is.
    image_height : int
        the number of pixels high the image is.
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
    coords_it = product(range(image_height), range(image_width))
    while True:
        out = list(islice(coords_it, batchsize))
        if len(out) == 0:
            return
        else:
            #  reversed on purpose so we get row-major indexing
            coords_y, coords_x = zip(*out)
            cx = np.array(coords_x)
            cy = np.array(coords_y)
            yield cx, cy
