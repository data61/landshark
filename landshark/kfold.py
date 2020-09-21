"""Low-ish memory cross validation indices."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterator

import numpy as np

BATCH_SIZE = 10000


def _batch_randn(
    start: int, stop: int, size: int, batch_size: int, seed: int
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
