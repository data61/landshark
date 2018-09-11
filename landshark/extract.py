"""Extracting utilities."""
import logging
from typing import List, Optional, Set, Tuple

import numpy as np

from landshark.fileio import parse_withlist
from landshark.metadata import (CategoricalMetadata, FeatureSetMetadata,
                                ContinuousMetadata)

log = logging.getLogger(__name__)


def get_active_features(feature_metadata: FeatureSetMetadata,
                        withfeat: List[str],
                        withoutfeat: List[str],
                        withlist: Optional[str]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    if len(withfeat) > 0 and len(withoutfeat) > 0:
        raise ValueError("Cant specificy withfeat and withoutfeat "
                         "at the same time")
    if withlist is not None and (len(withfeat) > 0 or len(withoutfeat) > 0):
        raise ValueError("Can't specify a feature list and command line "
                         "feature additions or subtractions")

    all_features: Set[str] = set()
    ncats = 0
    nords = 0
    if feature_metadata.continuous is not None:
        all_features = all_features.union(set(feature_metadata.continuous.labels))
        nords = len(feature_metadata.continuous.labels)
    if feature_metadata.categorical is not None:
        all_features = all_features.union(
            set(feature_metadata.categorical.labels))
        ncats = len(feature_metadata.categorical.labels)
    if withlist is not None:
        feature_list = parse_withlist(withlist)
    elif len(withfeat) > 0:
        feature_list = withfeat
    elif len(withoutfeat) > 0:
        feature_list = list(all_features.difference(set(withoutfeat)))
    else:
        feature_list = list(all_features)
    feature_set = set(feature_list)
    if not feature_set.issubset(all_features):
        print("Error, the following requested features do not appear "
              " in the data:\n{}\n Possible features are:\n{}".format(
                  set(feature_list).difference(all_features), all_features))
        raise ValueError("Requested features not in data")

    con_array = np.zeros(nords, dtype=bool)
    cat_array = np.zeros(ncats, dtype=bool)
    for f in feature_set:
        if feature_metadata.continuous is not None:
            try:
                idx = feature_metadata.continuous.labels.index(f)
                con_array[idx] = 1
            except ValueError:
                pass
        if feature_metadata.categorical is not None:
            try:
                idx = feature_metadata.categorical.labels.index(f)
                cat_array[idx] = 1
            except ValueError:
                pass

    log.info("Selecting {} of {} continuous features".format(
        np.sum(con_array), nords))
    log.info("Selecting {} of {} categorical features".format(
        np.sum(cat_array), ncats))
    return con_array, cat_array


def _subset_con_meta(m: ContinuousMetadata, active_cons: np.ndarray) \
        -> ContinuousMetadata:
    N = m.N
    D = np.sum(active_cons.astype(int))
    labels = [l for l, f in zip(m.labels, active_cons) if f]
    missing = m.missing
    means = m.means[active_cons] if m.means is not None else None
    variances = m.variances[active_cons] if m.variances is not None else None
    new_m = ContinuousMetadata(N, D, labels, missing, means, variances)
    return new_m


def _subset_cat_meta(m: CategoricalMetadata, active_cats: np.ndarray) \
        -> CategoricalMetadata:
    N = m.N
    D = np.sum(active_cats.astype(int))
    labels = [l for l, f in zip(m.labels, active_cats) if f]
    missing = m.missing
    ncategories = m.ncategories[active_cats]
    mappings = [e for e, f in zip(m.mappings, active_cats) if f]
    counts = [e for e, f in zip(m.counts, active_cats) if f]
    new_m = CategoricalMetadata(N, D, labels, missing, ncategories,
                                mappings, counts)
    return new_m


def active_column_metadata(m: FeatureSetMetadata, active_cons: np.ndarray,
                           active_cats: np.ndarray) -> FeatureSetMetadata:
    new_continuous: Optional[ContinuousMetadata] = None
    new_categorical: Optional[CategoricalMetadata] = None
    if m.continuous is not None and len(active_cons) > 0:
        new_continuous = _subset_con_meta(m.continuous, active_cons)
    if m.categorical is not None and len(active_cats) > 0:
        new_categorical = _subset_cat_meta(m.categorical, active_cats)

    new_m = FeatureSetMetadata(new_continuous, new_categorical, m.image)
    return new_m
