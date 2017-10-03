"""Patch extraction for images."""

import logging
from collections import namedtuple

import numpy as np
from typing import Tuple, List

log = logging.getLogger(__name__)


PatchRowRW = namedtuple("PatchRowRW", (["idx", "x", "y", "xp", "yp"]))
PatchMaskRowRW = namedtuple("PatchMaskRowRW", (["idx", "xp", "yp"]))


def patches(x_coords: np.ndarray, y_coords: np.ndarray, halfwidth: int,
            image_width: int, image_height: int) \
        -> Tuple[List[PatchRowRW], List[PatchMaskRowRW]]:
    """
    Generate the Read and write ops for patches given a set of coords.

    This function describes read and write operations in terms of a single
    index in y (row) followed by a contiguous slice in x. Patches are made
    up of many of these structs (PatchRowRW) (1 for each row in the patch).
    The output gives the read location in the image, and the write location
    in the patch array.

    The function also outputs the write operations for the image *mask*,
    ie what writes should be done on a 'False' mask to turn missing values into
    true. This is generally much more efficient that using the patch writes
    to write false into a 'True' mask.


    Parameters
    ----------
    x_coords : np.ndarray
        The x coordinates of the patch centres. Must be 1d and equal in size
        to y_coords.
    y_coords : np.ndarray
        The y coordinates of teh patch centres. Must be 1d and equal in size
        to the y_coords.
    halfwidth : int
        Integer describing the number of pixels out from the centre the patch
        should extend. A 1x1 patch has halfwidth 0. A 3x3 patch has halfwidth
        1 etc.
    image_width : int
        The width of the image in pixels. Needed for masking calculations.
    image_height : int
        The height of the image in pixels. Needed for masking calculations.

    Returns
    -------
    result : Tuple[List[PatchRowRW], List[PatchMaskRowRW]]
        The list of patch row read/writes and mask writes corresponding to
        the patches requested.

    """
    assert x_coords.shape[0] == y_coords.shape[0]
    assert x_coords.ndim == 1
    assert y_coords.ndim == 1
    assert halfwidth > 0
    assert image_width > 0

    ncoords = x_coords.shape[0]
    xmins = x_coords - halfwidth
    ymins = y_coords - halfwidth
    n = halfwidth * 2 + 1
    xmaxs = xmins + n - 1  # INCLUSIVE

    # What lines to read?
    y_reads = (ymins[np.newaxis, :] + np.arange(n)[:, np.newaxis]).flatten()
    patch_indices = np.tile(np.arange(ncoords), n)
    order = np.lexsort((patch_indices, y_reads))

    y_reads_sorted = y_reads[order]
    patch_indices_sorted = patch_indices[order]

    patch_rws = _patch_reads(n, y_reads_sorted, xmins, xmaxs, ymins,
                             patch_indices_sorted, image_width, image_height)
    mask_ws = _mask_patches(n, y_reads_sorted, xmins, xmaxs, ymins,
                            patch_indices_sorted, image_width, image_height)
    return patch_rws, mask_ws


def _patch_reads(n: int, y_reads: np.ndarray, xmins: np.ndarray,
                 xmaxs: np.ndarray, ymins: np.ndarray,
                 patch_indices: np.ndarray, image_width: int,
                 image_height: int) -> List[PatchRowRW]:
    """Compute the read and writes for the patches."""
    y_mask = np.logical_and(y_reads >= 0, y_reads < image_height)

    x_starts = np.maximum(xmins, 0)
    x_stops = np.minimum(xmaxs + 1, image_width)

    # patch space
    y_patch_reads = y_reads - ymins[patch_indices]
    x_patch_starts = x_starts - xmins
    x_patch_stops = x_stops - xmins

    patch_rw_list = []
    for i, m, y, yp in zip(patch_indices, y_mask, y_reads, y_patch_reads):
        if m:
            r = PatchRowRW(i, slice(x_starts[i], x_stops[i]), y,
                           slice(x_patch_starts[i], x_patch_stops[i]), yp)
            patch_rw_list.append(r)
    return patch_rw_list


def _mask_patches(n: int, y_reads: np.ndarray, xmins: np.ndarray,
                  xmaxs: np.ndarray, ymins: np.ndarray,
                  patch_indices: np.ndarray, image_width: int,
                  image_height: int) -> List[PatchMaskRowRW]:
    """Compute the inverse writes for the mask for the patches."""
    # Inverse (mask) writes
    inv_y_mask = np.logical_or(y_reads < 0, y_reads >= image_height)
    x_premask = xmins < 0
    x_postmask = xmaxs >= image_width
    y_patch_reads = y_reads - ymins[patch_indices]

    # There can be two x writes in general: pre- and post-image.
    x_patch_prestarts = np.zeros_like(xmins, dtype=int)
    x_patch_prestops = -1 * xmins
    x_patch_poststarts = np.full(xmins.shape, image_width) - xmins
    x_patch_poststops = (xmaxs + 1) - xmins

    mask_w_list = []
    for i, m, yp in zip(patch_indices, inv_y_mask, y_patch_reads):
        if m:
            mask_w_list.append(PatchMaskRowRW(i, slice(0, n), yp))
        else:
            if x_premask[i]:
                mask_w_list.append(PatchMaskRowRW(i,
                                   slice(x_patch_prestarts[i],
                                         x_patch_prestops[i]), yp))
            if x_postmask[i]:
                mask_w_list.append(PatchMaskRowRW(i,
                                   slice(x_patch_poststarts[i],
                                         x_patch_poststops[i]), yp))
    return mask_w_list
