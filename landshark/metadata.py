"""Metadata."""

import os.path
import pickle
import numpy as np
from typing import NamedTuple, List, Optional, cast, Union, Dict, Any

import numpy as np

from landshark.image import ImageSpec
from landshark.basetypes import OrdinalType, CategoricalType


class Array2DMetadata:
    def __init__(self, N: int, D: int, labels: List[str],
                 missing: Optional[Union[OrdinalType, CategoricalType]]) -> None:
        self.N = N
        self.D = D
        self.labels = labels
        self.missing = missing

class OrdinalMetadata(Array2DMetadata):
    def __init__(self, N: int, D: int, labels: List[str],
                 missing: Optional[OrdinalType], means: Optional[np.ndarray],
                 variances: Optional[np.ndarray]) -> None:
        super().__init__(N, D , labels, missing)
        self.means = means
        self.variances = variances

class CategoricalMetadata(Array2DMetadata):
    def __init__(self, N: int, D: int, labels: List[str],
                 missing: Optional[CategoricalType], ncategories: np.ndarray,
                 mappings: np.ndarray, counts: np.ndarray) -> None:
        super().__init__(N, D, labels, missing)
        self.ncategories = ncategories
        self.mappings = mappings
        self.counts = counts

class FeatureSetMetadata:

    def __init__(self,
                 ordinal: Optional[OrdinalMetadata],
                 categorical: Optional[CategoricalMetadata],
                 image: ImageSpec) -> None:
        assert not(ordinal is None and categorical is None)
        if ordinal and not categorical:
            self.N = ordinal.N
        elif categorical and not ordinal:
            self.N = categorical.N
        elif categorical and ordinal:
            assert ordinal.N == categorical.N
            self.N = ordinal.N
        else:
            raise ValueError("Must have at least 1 of ordinal or categorical")
        self.ordinal = ordinal
        self.categorical = categorical
        self.image = image


TargetMetadata = Union[OrdinalMetadata, CategoricalMetadata]

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
