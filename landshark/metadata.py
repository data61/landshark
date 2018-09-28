"""Metadata."""

import os.path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Union, cast

import numpy as np

from landshark.basetypes import CategoricalType, ContinuousType
from landshark.image import ImageSpec

class PickleObj:

    _filename = None

    @classmethod
    def load(cls, directory):
        path = os.path.join(directory, _filename + ".bin")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save(self, directory: str) -> None:
        path = os.path.join(directory, _filename + ".bin")
        with open(spec_path, "wb") as f:
            pickle.dump(self, f)


class CategoricalFeature(NamedTuple):
    nvalues: int
    D: int
    mapping: np.ndarray
    counts: np.ndarray

class ContinuousFeature(NamedTuple):
    D: int
    mean: np.ndarray
    sd: np.ndarray


class ContinuousFeatureSet:

    def __init__(self, labels, missing, means, variances) -> None:

        D = len(labels)
        if means is None:
            means = [None] * D
            variances = [None] * D

        self._missing = missing
        self._columns = {l: ContinuousFeature(1, m, v)
            for l, m, v in zip(labels, means, variances)}
        self._n = len(self._columns)

    @property
    def columns(self):
        return self._columns

    @property
    def missing_value(self):
        return self._missing

    def __len__(self):
        return self._n


class CategoricalFeatureSet:

    def __init__(self, labels, missing, nvalues, mappings, counts) \
                 -> None:
        self._missing = missing
        self._columns = {l: CategoricalFeature(n, 1, m, c)
            for l, n, m, c in zip(labels, nvalues, mappings, counts)}
        self._n = len(self._columns)

    @property
    def columns(self):
        return self._columns

    @property
    def missing_value(self):
        return self._missing

    def __len__(self):
        return self._n


class FeatureSet(PickleObj):

    _filename = "FEATURESET"

    def __init__(self, continuous: Optional[ContinuousFeatureSet],
                 categorical: Optional[CategoricalFeatureSet],
                 image: ImageSpec, N: int) -> None:
        self.continuous = continuous
        self.categorical = categorical
        self.image = image
        self._N = N

    def __len__(self):
        return self._N


class CategoricalTarget(PickleObj):

    _filename = "CATEGORICALTARGET"

    def __init__(self, N, labels, nvalues: np.ndarray,
                 mappings: List[np.ndarray], counts: List[np.ndarray]) \
            -> None:
        self.N = N
        self.D = len(labels)
        self.nvalues = nvalues
        self.mappings = mappings
        self.counts = counts
        self.labels = labels

class ContinuousTarget(PickleObj):

    _filename = "CONTINUOUSTARGET"

    def __init__(self, N: int, labels, means: np.ndarray,
                 variances: np.ndarray) -> None:
        self.N = N
        self.D = len(labels)
        self.means = means
        self.variances = variances
        self.labels = labels


Target = Union[ContinuousTarget, CategoricalTarget]


class Training(PickleObj):

    _filename = "TRAINING"

    def __init__(self, targets: Target, features: FeatureSet, nfolds: int,
                 testfold: int, fold_counts: Dict[int, int]) -> None:
        self.targets = targets
        self.features = features
        self.nfolds = nfolds
        self.testfold = testfold
        self.found_counts = found_counts
