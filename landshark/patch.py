"""Patch extraction for images."""

import logging

import numpy as np

log = logging.getLogger(__name__)


class PatchRange:
    """
    Fixed-y xrange for contiguous patch strips.

    Parameters
    ----------
    y : int
        The y index of the strip
    xstart : int
        The start of the x range of the strip
    xend : int
        The exclusive end indeex, so array[xstart:xend] is correct.

    """

    def __init__(self, y: int, xstart: int, xstop: int) -> None:
        """Create a PatchRange object."""
        self.y = y
        self.x = slice(xstart, xstop)


class PatchRead:
    """
    An abstraction of the patch reading operation.

    Contains two slices of the same size, one to fill the patch,
    and one to index the image.

    Parameters
    ----------
    y : int
        The y coordinate of the read
    xstart : int
        The start of the x coordinate of the read
    xstop : int
        The exclusive end x coordinate of the read
    x_offset : int
        The x offset of the patch in pixels (in the image).
    y_offset : int
        The y offset of the patch in pixels (in the image).

    """

    def __init__(self, y: int, xstart: int, xstop: int, x_offset: int,
                 y_offset: int) -> None:
        """Create a PatchRead object."""
        self.image = PatchRange(y, xstart, xstop)
        self.patch = PatchRange(y - y_offset, xstart - x_offset,
                                xstop - x_offset)


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
        mask = np.ones((n, n), dtype=bool)
        # What lines to read?
        ystart = max(0, ymin)
        xstart = max(0, xmin)
        ystop = min(image_height, ymax + 1)  # EXCLUSIVE
        xstop = min(image_width, xmax + 1)  # EXCLUSIVE

        y_lines = range(ystart, ystop)
        reads = [PatchRead(y, xstart, xstop, xmin, ymin) for y in y_lines]

        # create the mask
        for r in reads:
            mask[r.patch.y, r.patch.x] = False

        self.reads = reads
        self.mask = mask
