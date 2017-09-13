"""Patch extraction for images."""

import logging

import numpy as np

log = logging.getLogger(__name__)


class Patch:
    """
    Contains all the information required to read a patch.

    contains the patch-space indices and image-space indices for all required
    reads, and the associated nodata mask.

    Parameters
    ----------
    x_centre : int
        The horizontal index of the patch centre pixel.
    y_centre : int
        The vertical index of the patch centre pixel.
    halfwidth : int
        The number of pixels from the centre pixel to size the patch. For
        example, halfwidth=1 creates a 3x3 patch.
    image_width : int
        The number of pixels horizontally in the image
    image_height : int
        The number of pixels vertically in the image.

    """

    def __init__(self, x_centre: int, y_centre: int, halfwidth: int,
                 image_width: int, image_height: int) -> None:
        """Construct a patch object."""
        xmin = x_centre - halfwidth
        ymin = y_centre - halfwidth
        n = halfwidth * 2 + 1
        ymax = ymin + n - 1  # INCLUSIVE
        xmax = xmin + n - 1  # INCLUSIVE
        # What lines to read?
        ystart = max(0, ymin)
        xstart = max(0, xmin)
        ystop = min(image_height, ymax + 1)  # EXCLUSIVE
        xstop = min(image_width, xmax + 1)  # EXCLUSIVE

        self.y_indices = np.arange(ystart, ystop)

        self.patch_y_indices = self.y_indices - ymin
        self.x = slice(xstart, xstop)
        self.patch_x = slice(xstart - xmin, xstop - xmin)

        patch_flat = []
        flat = []

        self.mask = np.ones((n, n), dtype=bool)
        for y in self.patch_y_indices:
            self.mask[y, self.patch_x] = False

        #  Create data structures for flattened masks and images
        self.flat_mask = self.mask.flatten("C")
        for y, yp in zip(self.y_indices, self.patch_y_indices):
            start = y * image_width + xstart
            stop = y * image_width + xstop
            flat.append(slice(start, stop))
            pstart = yp * n + (xstart - xmin)
            pstop = yp * n + (xstop - xmin)
            patch_flat.append(slice(pstart, pstop))
        self.patch_flat = patch_flat
        self.flat = flat
