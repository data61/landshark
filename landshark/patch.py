"""Patch extraction for images."""

import logging
from collections import namedtuple

import numpy as np

log = logging.getLogger(__name__)


PatchRowRead = namedtuple("PatchRowRead", (["idx", "x", "y", "xp", "yp"]))


def patches(x_coords, y_coords, halfwidth, image_width, image_height):
    """Generate patches."""
    ncoords = x_coords.shape[0]
    xmins = x_coords - halfwidth
    ymins = y_coords - halfwidth
    n = halfwidth * 2 + 1
    xmaxs = xmins + n - 1  # INCLUSIVE

    # What lines to read?
    y_reads = (ymins[np.newaxis, :] + np.arange(n)[:, np.newaxis]).flatten()
    patch_idx = np.tile(np.arange(ncoords), n)
    order = np.lexsort((patch_idx, y_reads))

    y_reads = y_reads[order]
    patch_idx = patch_idx[order]
    y_mask = np.logical_and(y_reads >= 0, y_reads < image_height)

    # TODO optimise
    x_starts = np.maximum(xmins, 0)
    x_stops = np.minimum(xmaxs + 1, image_width)

    # patch space
    y_patch_reads = y_reads - ymins[patch_idx]
    x_patch_starts = x_starts - xmins
    x_patch_stops = x_stops - xmins

    for i, m, y, yp in zip(patch_idx, y_mask, y_reads, y_patch_reads):
        if m:
            yield PatchRowRead(i, slice(x_starts[i], x_stops[i]), y,
                               slice(x_patch_starts[i], x_patch_stops[i]), yp)
