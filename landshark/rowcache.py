"""Row cache for storing parts of image stacks in memory."""

import logging

import numpy as np
from lru import LRU
from tables.carray import CArray

log = logging.getLogger("__name__")


class RowCache:
    """
    Least-Recently-Used (LRU) cache for blocks of image rows.

    This class stores a pre-configured number of 'row blocks' in memory,
    removing the least-recently used block when the user requests a block
    not currently in memory.

    Parameters
    ----------
    carray : CArray
        The HDF5 CArray object containing the image data.
    rows_per_block : int
        The (maximum) number of rows in each block. The final block may be
        smaller than this.
    nblocks : int
        The (requested) number of blocks to store in memory at one time.
        If the block size is large, then we may have fewer blocks that take
        up the whole image.

    """

    def __init__(self, harray: CArray, rows_per_block: int,
                 nblocks: int) -> None:
        """Create an instance of the RowCache object."""
        self._harray = harray
        self._d = LRU(nblocks)
        self.total_rows = harray.shape[0]
        self.rows_per_block = rows_per_block
        total_blocks = (self.total_rows // rows_per_block) + \
            int(self.total_rows % rows_per_block != 0)
        if (total_blocks < nblocks):
            log.warning("Image size constraints mean "
                        "{} blocks defined for image but {} requested to be "
                        "cached".format(total_blocks, nblocks))
        self._nblocks = int(min(total_blocks, nblocks))
        block_starts = np.arange(total_blocks, dtype=int) * rows_per_block
        block_stops = block_starts + rows_per_block
        block_stops[-1] = self.total_rows

        self.block_slices = [slice(i, j) for i, j in
                             zip(block_starts, block_stops)]

        # fill the buffer initially
        log.info("Pre-filling the row cache with {} blocks".format(
            self._nblocks))
        for i in range(self._nblocks):
            self._d[i] = self._harray[self.block_slices[i]]

    @property
    def nblocks(self) -> int:
        """
        Get the number of blocks stored in the cache.

        If the blocks are large, this might be lower than the number requested.

        Returns
        -------
        result : int
            The number of blocks that the cache stores.

        """
        return self._nblocks

    def __call__(self, idx: int, xslice: slice) -> np.ndarray:
        """
        Get a slice of data from the cache.

        Parameters
        ----------
        idx : int
            The row (y) index of the slice.
        xslice: slice
            The slice in x over the row specificed by idx.

        Returns
        -------
        d : np.ndarray
            The contiguous block of data from that location in the image.

        """
        b = idx // self.rows_per_block
        if b not in self._d:
            self._d[b] = self._harray[self.block_slices[b]]
        in_block_idx = idx - (b * self.rows_per_block)
        d = self._d[b][in_block_idx, xslice]
        return d
