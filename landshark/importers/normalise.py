import numpy as np

class _Statistics:
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

