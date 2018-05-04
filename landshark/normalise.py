import numpy as np
import logging

from tqdm import tqdm
from typing import Tuple, Optional

from landshark import iteration
from landshark.basetypes import OrdinalArraySource, Worker, OrdinalType
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
    def variance(self) -> np.ndarray:
        """Get the current estimate of the variance."""
        assert np.all(self._n > 1)
        var = self._m2 / self._n
        return var

    @property
    def count(self) -> np.ndarray:
        """Get the count of each feature."""
        return self._n


class Normaliser(Worker):

    def __init__(self, mean: np.ndarray, var: np.ndarray,
                 missing: Optional[OrdinalType]) -> None:
        self._mean = mean
        self._std = np.sqrt(var)
        self._missing = missing

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xm = to_masked(x, self._missing)
        xm -= self._mean
        xm /= self._std
        return xm.data


def get_stats(src: OrdinalArraySource, batchsize: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    log.info("Computing ordinal feature statistics")
    n_rows = src.shape[0]
    n_cols = src.shape[-1]
    stats = StatCounter(n_cols)
    with tqdm(total=n_rows) as pbar:
        with src:
            for s in iteration.batch_slices(batchsize, n_rows):
                x = src(s)
                bs = x.reshape((-1, x.shape[-1]))
                bm = to_masked(bs, src.missing)
                stats.update(bm)
                pbar.update(x.shape[0])
    mean, variance = stats.mean, stats.variance
    return mean, variance
