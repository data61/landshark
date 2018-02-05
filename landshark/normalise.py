import numpy as np

from landshark import iteration
from landshark.util import to_masked


class Normaliser:
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
        new_mean = np.ma.mean(array, axis=0)
        new_m2 = np.ma.var(array, axis=0, ddof=0) * new_n

        delta = new_mean - self._mean
        delta_mean = delta * (new_n / (new_n + self._n))

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


class NormaliserPreprocessor:

    def __init__(self, ncols, missing_values):
        self.ncols = ncols
        self.missing_values = missing_values

    def __call__(self, x):
        bs = x.reshape((-1, self.ncols))
        bm = to_masked(bs, self.missing_values)
        return bm


class OrdinalOutputTransform:
    def __init__(self, mean, variance, missing_values):
        self.mean = mean
        self.variance = variance
        self.missing_values = missing_values

    def __call__(self, x):
        bm = to_masked(x, self.missing_values)
        bm -= self.mean
        bm /= np.sqrt(self.variance)
        out = bm.data
        return out


def get_stats(array_src, batchsize, pool):
    n_rows = array_src.shape[0]
    n_features = array_src.shape[-1]
    missing_values = array_src.missing
    norm = Normaliser(n_features)
    it = iteration.batch_slices(batchsize, n_rows)
    f = NormaliserPreprocessor(n_features, missing_values)
    data_it = ((array_src.slice(start, end)) for start, end in it)
    out_it = pool.imap(f, data_it)
    for ma in out_it:
        norm.update(ma)
    return norm.mean, norm.variance



