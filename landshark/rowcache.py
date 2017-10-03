from lru import LRU

class RowCache:
    def __init__(self, harray, rows_per_block, nblocks):
        self._harray = harray
        self._d = LRU(nblocks)
        self.total_rows = harray.shape[0]
        self.rows_per_block = rows_per_block
        total_blocks = (self.total_rows // rows_per_block) + \
            int(self.total_rows % rows_per_block != 0)
        block_starts = np.arange(total_blocks, dtype=int) * rows_per_block
        block_stops = block_starts + rows_per_block
        block_stops[-1] = self.total_rows

        self.block_slices = [slice(i, j) for i, j in
                             zip(block_starts, block_stops)]

        # fill the buffer initially
        for i, s in enumerate(self.block_slices):
            self._d[i] = self._harray[s]

    def __call__(self, idx, xslice):
        b = idx // self.rows_per_block
        if b not in self._d:
            self._d[b] = self._harray[self.block_slices[b]]
        in_block_idx = idx - (b * self.rows_per_block)
        d = self._d[b][in_block_idx, xslice]
        return d

