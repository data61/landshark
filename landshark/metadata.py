"""Metadata."""

import os.path
import pickle
import tables
import numpy as np
from typing import NamedTuple, List, Optional, cast

from landshark.image import ImageSpec
from landshark.basetypes import OrdinalType, CategoricalType


class TrainingMetadata(NamedTuple):
    """Metadata that training alogrithms need to know."""

    ntargets: int
    target_dtype: np.dtype
    nfeatures_ord: int
    nfeatures_cat: int
    halfwidth: int
    N: int
    target_labels: List[str]
    image_spec: ImageSpec
    ncategories: Optional[List[int]]
    ncategories_patched: Optional[List[int]]
    target_map: Optional[np.ndarray]
    target_counts: Optional[List[List[int]]]
    folds: int
    testfold: int
    missing_ord: Optional[OrdinalType]
    missing_cat: Optional[CategoricalType]


def from_files(feature_file: str,
               target_file: str,
               image_spec: ImageSpec,
               halfwidth: int,
               n_train: int,
               folds: int,
               testfold: int) -> TrainingMetadata:
    """TODO."""
    nfeatures_ord = 0
    nfeatures_cat = 0
    ncategories = None
    ncategories_patched = None
    target_counts = None
    target_map = None
    missing_ord = None
    missing_cat = None
    with tables.open_file(feature_file, "r") as hfile:
        if hasattr(hfile.root, "categorical_data"):
            nfeatures_cat = hfile.root.categorical_data.atom.shape[0]
            ncategories = [len(k) for k in hfile.root.categorical_mappings]
            bmul = (2 * halfwidth + 1) ** 2
            # ncategories_patched = ncategories * bmul
            ncats_nested = [[k] * bmul for k in ncategories]
            ncategories_patched = [e for l in ncats_nested for e in l]
            missing_cat = hfile.root.categorical_data.attrs.missing

        if hasattr(hfile.root, "ordinal_data"):
            missing_ord = hfile.root.ordinal_data.attrs.missing
            nfeatures_ord = hfile.root.ordinal_data.atom.shape[0]

    with tables.open_file(target_file, "r") as hfile:
        # we know it has to have either categorical or ordinal
        if hasattr(hfile.root, "categorical_data"):
            ntargets = hfile.root.categorical_data.atom.shape[0]
            target_dtype = CategoricalType
            target_labels = [s.decode() for s in
                             hfile.root.categorical_data_columns.read()]
            target_counts = list(hfile.root.categorical_counts)
            target_map = list(hfile.root.categorical_mappings)
        elif hasattr(hfile.root, "ordinal_data"):
            target_labels = [s.decode() for s in
                             hfile.root.ordinal_data_columns.read()]
            ntargets = hfile.root.ordinal_data.atom.shape[0]
            target_dtype = OrdinalType
        else:
            raise ValueError("Target hfile does not have data")

    m = TrainingMetadata(ntargets=ntargets,
                         target_dtype=target_dtype,
                         nfeatures_ord=nfeatures_ord,
                         nfeatures_cat=nfeatures_cat,
                         halfwidth=halfwidth,
                         N=n_train,
                         ncategories=ncategories,
                         target_labels=target_labels,
                         image_spec=image_spec,
                         target_map=target_map,
                         target_counts=target_counts,
                         ncategories_patched=ncategories_patched,
                         folds=folds,
                         testfold=testfold,
                         missing_ord=missing_ord,
                         missing_cat=missing_cat
                         )
    return m


def write_metadata(directory: str, m: TrainingMetadata) -> None:
    """TODO."""
    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(m, f)


def load_metadata(path: str) -> TrainingMetadata:
    """TODO."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
        m = cast(TrainingMetadata, obj)  # no way to know type from pickle
    return m
