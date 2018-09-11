"""Metadata."""

import os.path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Union, cast

import numpy as np

from landshark.basetypes import CategoricalType, ContinuousType
from landshark.image import ImageSpec


class Array2DMetadata:
    def __init__(self, N: int, D: int, labels: List[str],
                 missing: Optional[Union[ContinuousType, CategoricalType]]
                 ) -> None:
        self.N = N
        self.D = D
        self.labels = labels
        self.missing = missing


class ContinuousMetadata(Array2DMetadata):
    def __init__(self, N: int, D: int, labels: List[str],
                 missing: Optional[ContinuousType], means: Optional[np.ndarray],
                 variances: Optional[np.ndarray]) -> None:
        super().__init__(N, D, labels, missing)
        self.means = means
        self.variances = variances


class CategoricalMetadata(Array2DMetadata):
    def __init__(self, N: int, D: int, labels: List[str],
                 missing: Optional[CategoricalType], ncategories: np.ndarray,
                 mappings: List[np.ndarray], counts: List[np.ndarray]) -> None:
        super().__init__(N, D, labels, missing)
        self.ncategories = ncategories
        self.mappings = mappings
        self.counts = counts


class FeatureSetMetadata:

    def __init__(self,
                 continuous: Optional[ContinuousMetadata],
                 categorical: Optional[CategoricalMetadata],
                 image: ImageSpec) -> None:
        assert not(continuous is None and categorical is None)
        if continuous and not categorical:
            self.N = continuous.N
        elif categorical and not continuous:
            self.N = categorical.N
        elif categorical and continuous:
            assert continuous.N == categorical.N
            self.N = continuous.N
        else:
            raise ValueError("Must have at least 1 of continuous or categorical")
        self.continuous = continuous
        self.categorical = categorical
        self.image = image
        self.D_continuous = continuous.D if continuous else 0
        self.D_categorical = categorical.D if categorical else 0


TargetMetadata = Union[ContinuousMetadata, CategoricalMetadata]


class TrainingMetadata(NamedTuple):
    targets: TargetMetadata
    features: FeatureSetMetadata
    halfwidth: int
    nfolds: int
    testfold: int
    fold_counts: Dict[int, int]


class QueryMetadata(NamedTuple):
    features: FeatureSetMetadata


def pickle_metadata(directory: str, m: Any) -> None:
    """TODO."""
    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(m, f)


def _load_metadata(path: str) -> Any:
    """TODO."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def unpickle_training_metadata(path: str) -> TrainingMetadata:
    obj = _load_metadata(path)
    m = cast(TrainingMetadata, obj)
    return m


def unpickle_query_metadata(path: str) -> QueryMetadata:
    obj = _load_metadata(path)
    m = cast(QueryMetadata, obj)
    return m
