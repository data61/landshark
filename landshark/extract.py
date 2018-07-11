"""Extracting utilities."""
import logging
from landshark.metadata import FeatureSetMetadata, OrdinalMetadata,\
    CategoricalMetadata
from landshark.fileio import parse_withlist
from typing import Set, List, Tuple, Optional
import numpy as np

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
    if feature_metadata.ordinal is not None:
        all_features = all_features.union(set(feature_metadata.ordinal.labels))
        nords = len(feature_metadata.ordinal.labels)
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

    ord_array = np.zeros(nords, dtype=bool)
    cat_array = np.zeros(ncats, dtype=bool)
    for f in feature_set:
        if feature_metadata.ordinal is not None:
            try:
                idx = feature_metadata.ordinal.labels.index(f)
                ord_array[idx] = 1
            except ValueError:
                pass
        if feature_metadata.categorical is not None:
            try:
                idx = feature_metadata.categorical.labels.index(f)
                cat_array[idx] = 1
            except ValueError:
                pass

    log.info("Selecting {} of {} ordinal features".format(
        np.sum(ord_array), nords))
    log.info("Selecting {} of {} categorical features".format(
        np.sum(cat_array), ncats))
    return ord_array, cat_array


def _subset_ord_meta(m: OrdinalMetadata, active_ords: np.ndarray) \
        -> OrdinalMetadata:
    N = m.N
    D = np.sum(active_ords.astype(int))
    labels = [l for l, f in zip(m.labels, active_ords) if f]
    missing = m.missing
    means = m.means[active_ords] if m.means is not None else None
    variances = m.variances[active_ords] if m.variances is not None else None
    new_m = OrdinalMetadata(N, D, labels, missing, means, variances)
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


def active_column_metadata(m: FeatureSetMetadata, active_ords: np.ndarray,
                           active_cats: np.ndarray) -> FeatureSetMetadata:
    new_ordinal: Optional[OrdinalMetadata] = None
    new_categorical: Optional[CategoricalMetadata] = None
    if m.ordinal is not None and len(active_ords) > 0:
        new_ordinal = _subset_ord_meta(m.ordinal, active_ords)
    if m.categorical is not None and len(active_cats) > 0:
        new_categorical = _subset_cat_meta(m.categorical, active_cats)

    new_m = FeatureSetMetadata(new_ordinal, new_categorical, m.image)
    return new_m
