"""Image operations that move between world and image coordinates."""
import logging

import numpy as np
from affine import Affine
# mypy type checking
from typing import Tuple

log = logging.getLogger(__name__)


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
    pixel_height = -1.0 * affine[4]  # origin is TOP left
    origin_x = affine[2]
    origin_y = affine[5]

    # construct the canonical pixel<->position map: +1 for outer corner
    pix_x = np.arange(width + 1, dtype=np.float64)
    pix_y = np.arange(height + 1, dtype=np.float64)
    coords_x = (pix_x * pixel_width) + origin_x
    coords_y = (pix_y * pixel_height) + origin_y

    return coords_x, coords_y
