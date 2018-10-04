"""Metadata."""

from collections import OrderedDict
import os.path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Union, cast

import numpy as np

from landshark.basetypes import CategoricalType, ContinuousType
from landshark.image import ImageSpec

class PickleObj:

    _filename: Optional[str] = None

    @classmethod
    def load(cls, directory: str) -> Any:
        if not cls._filename:
            raise NotImplementedError("PickleObj must be subclassed")
        path = os.path.join(directory, cls._filename)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save(self, directory: str) -> None:
        if not self._filename:
            raise NotImplementedError("PickleObj must be subclassed")
        path = os.path.join(directory, self._filename)
        with open(path, "wb") as f:
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


Feature = Union[CategoricalFeature, ContinuousFeature]

class ContinuousFeatureSet:

    def __init__(self, labels: List[str], missing: ContinuousType,
                 means: np.ndarray, variances: np.ndarray) -> None:

        D = len(labels)
        if means is None:
            means = [None] * D
            variances = [None] * D

        self._missing = missing
        self._columns = OrderedDict([(l, ContinuousFeature(1, m, v))
            for l, m, v in zip(labels, means, variances)])
        self._n = len(self._columns)

    @property
    def columns(self) -> OrderedDict:
        return self._columns

    @property
    def missing_value(self) -> ContinuousType:
        return self._missing

    def __len__(self) -> int:
        return self._n


class CategoricalFeatureSet:

    def __init__(self, labels: List[str], missing: CategoricalType,
                 nvalues: np.ndarray, mappings: List[np.ndarray],
                 counts: np.ndarray) -> None:
        self._missing = missing
        self._columns = OrderedDict([(l, CategoricalFeature(n, 1, m, c))
            for l, n, m, c in zip(labels, nvalues, mappings, counts)])
        self._n = len(self._columns)

    @property
    def columns(self) -> OrderedDict:
        return self._columns

    @property
    def missing_value(self) -> CategoricalType:
        return self._missing

    def __len__(self) -> int:
        return self._n


class FeatureSet(PickleObj):

    _filename = "FEATURESET.bin"

    def __init__(self, continuous: Optional[ContinuousFeatureSet],
                 categorical: Optional[CategoricalFeatureSet],
                 image: ImageSpec, N: int, halfwidth: int) -> None:
        self.continuous = continuous
        self.categorical = categorical
        self.image = image
        self._N = N
        self.halfwidth = halfwidth

    def __len__(self) -> int:
        return self._N


class CategoricalTarget(PickleObj):

    _filename = "CATEGORICALTARGET.bin"
    dtype = CategoricalType

    def __init__(self, N: int, labels: np.ndarray, nvalues: np.ndarray,
                 mappings: List[np.ndarray], counts: List[np.ndarray]) \
            -> None:
        self.N = N
        self.D = len(labels)
        self.nvalues = nvalues
        self.mappings = mappings
        self.counts = counts
        self.labels = labels

class ContinuousTarget(PickleObj):

    _filename = "CONTINUOUSTARGET.bin"
    dtype = ContinuousType

    def __init__(self, N: int, labels: np.ndarray, means: np.ndarray,
                 variances: np.ndarray) -> None:
        self.N = N
        self.D = len(labels)
        self.means = means
        self.variances = variances
        self.labels = labels


Target = Union[ContinuousTarget, CategoricalTarget]


class Training(PickleObj):

    _filename = "TRAINING.bin"

    def __init__(self, targets: Target, features: FeatureSet, nfolds: int,
                 testfold: int, fold_counts: Dict[int, int]) -> None:
        self.targets = targets
        self.features = features
        self.nfolds = nfolds
        self.testfold = testfold
        self.fold_counts = fold_counts
