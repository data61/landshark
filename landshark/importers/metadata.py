
"""Metadata."""

import os.path
import pickle
from collections import namedtuple

class TrainingMetadata:
    def __init__(self, ntargets, target_dtype, nfeatures_ord, nfeatures_cat,
                 halfwidth, N, ncategories, target_labels, image_spec,
                 target_map, target_counts):
        self.ntargets = ntargets
        self.target_dtype = target_dtype
        self.nfeatures_ord = nfeatures_ord
        self.nfeatures_cat = nfeatures_cat
        self.halfwidth = halfwidth
        self.N = N
        self.ncategories = ncategories
        self.target_labels = target_labels
        self.image_spec = image_spec
        self.target_map = target_map
        self.target_counts = target_counts


def from_data(feature_obj, target_obj, halfwidth, n_train):

    if feature_obj.ord is None:
        nfeatures_ord = 0
    else:
        nfeatures_ord = feature_obj.ord.nfeatures

    if feature_obj.cat is None:
        nfeatures_cat = 0
        ncategories = 0
    else:
        nfeatures_cat = feature_obj.cat.nfeatures
        ncategories = feature_obj.cat.ncategories

    m = TrainingMetadata(ntargets=len(target_obj.labels),
                         target_dtype = target_obj.dtype,
                         nfeatures_ord = nfeatures_ord,
                         nfeatures_cat = nfeatures_cat,
                         halfwidth=halfwidth,
                         N=n_train,
                         ncategories=ncategories,
                         target_labels=target_obj.labels,
                         image_spec=feature_obj.image_spec,
                         target_map=target_obj.categorical_map,
                         target_counts=target_obj.categorical_counts)
    return m

def write_metadata(directory, m):
    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(m, f)


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

