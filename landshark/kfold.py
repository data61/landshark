"""Low-ish memory cross validation indices."""
from typing import Iterator

import numpy as np

BATCH_SIZE = 10000


def _batch_randn(start: int,
                 stop: int,
                 size: int,
                 batch_size: int,
                 seed: int
                 ) -> Iterator[np.ndarray]:
    rnd = np.random.RandomState(seed)
    total_n = 0
    while total_n < size:
        batch_start = total_n
        batch_end = min(total_n + batch_size, size)
        batch_n = batch_end - batch_start
        vals = rnd.randint(start, stop, size=(batch_n))
        yield vals
        total_n += batch_n
    return


class KFolds:

    def __init__(self, N: int, K: int = 10, seed: int = 666) -> None:
        """Low-ish memory k-fold cross validation indices generator.

        Args:
            N (int): Number of samples.
            K (int, optional): Defaults to 10. Number of folds.
            seed (int, optional): Defaults to 666. Random seed.
        """
        self.K = K
        self.N = N
        self.seed = seed
        self.counts = {k: 0 for k in range(1, self.K + 1)}

        for vals in _batch_randn(1, K + 1, N, BATCH_SIZE, self.seed):
            indices, counts = np.unique(vals, return_counts=True)
            for k, v in zip(indices, counts):
                self.counts[k] += v

    def iterator(self, batch_size: int) -> Iterator[np.ndarray]:
        """Return an iterator of fold index batches."""
        return _batch_randn(1, self.K + 1, self.N, batch_size, self.seed)
