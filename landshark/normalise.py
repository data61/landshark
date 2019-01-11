"""Classes for normalising imput data prior to running a model."""

import logging
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from landshark import iteration
from landshark.basetypes import ContinuousArraySource, ContinuousType, Worker
from landshark.util import to_masked

log = logging.getLogger(__name__)


class StatCounter:
    """Class that computes online mean and variance."""

    def __init__(self, n_features: int) -> None:
        """Initialise the counters."""
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._n = np.zeros(n_features, dtype=int)

    def update(self, array: np.ma.MaskedArray) -> None:
        """Update calclulations with new data."""
        assert array.ndim == 2
        assert array.shape[0] > 1

        new_n = np.ma.count(array, axis=0)
        new_mean = (np.ma.mean(array, axis=0)).data
        new_mean[new_n == 0] = 0.  # enforce this condition
        new_m2 = (np.ma.var(array, axis=0, ddof=0) * new_n).data

        add_n = new_n + self._n
        if any(add_n == 0):  # catch any totally masked images
            add_n[add_n == 0] = 1

        delta = new_mean - self._mean
        delta_mean = delta * (new_n / add_n)

        self._mean += delta_mean
        self._m2 += new_m2 + (delta * self._n * delta_mean)
        self._n += new_n

    @property
    def mean(self) -> np.ndarray:
        """Get the current estimate of the mean."""
        assert np.all(self._n > 1)
        return self._mean

    @property
    def sd(self) -> np.ndarray:
        """Get the current estimate of the standard deviation."""
        assert np.all(self._n > 1)
        var = self._m2 / self._n
        sd = np.sqrt(var)
        return sd

    @property
    def count(self) -> np.ndarray:
        """Get the count of each feature."""
        return self._n


class Normaliser(Worker):

    def __init__(self,
                 mean: np.ndarray,
                 sd: np.ndarray,
                 missing: Optional[ContinuousType]
                 ) -> None:
        self._mean = mean
        self._sd = sd
        self._missing = missing

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xm = to_masked(x, self._missing)
        xm -= self._mean
        xm /= self._sd
        return xm.data


def get_stats(src: ContinuousArraySource,
              batchrows: int
              ) -> Tuple[np.ndarray, np.ndarray]:
    log.info("Computing continuous feature statistics")
    n_rows = src.shape[0]
    n_cols = src.shape[-1]
    stats = StatCounter(n_cols)
    with tqdm(total=n_rows) as pbar:
        with src:
            for s in iteration.batch_slices(batchrows, n_rows):
                x = src(s)
                bs = x.reshape((-1, x.shape[-1]))
                bm = to_masked(bs, src.missing)
                stats.update(bm)
                pbar.update(x.shape[0])
    mean, sd = stats.mean, stats.sd
    return mean, sd
