
"""Metadata."""

import os.path
import pickle
import tables
from landshark.basetypes import OrdinalType, CategoricalType

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


def from_files(feature_file, target_file, image_spec, halfwidth, n_train):

    nfeatures_ord = 0
    nfeatures_cat = 0
    ncategories = None
    with tables.open_file(feature_file, "r") as hfile:
        if hasattr(hfile.root, "categorical_data"):
            nfeatures_cat = hfile.root.categorical_data.atom.shape[0]
            ncategories = [len(k) for k in hfile.root.categorical_mappings]

        if hasattr(hfile.root, "ordinal_data"):
            nfeatures_ord = hfile.root.ordinal_data.atom.shape[0]

    with tables.open_file(target_file, "r") as hfile:
        # we know it has to have either categorical or ordinal
        if hasattr(hfile.root, "categorical_data"):
            ntargets = hfile.root.categorical_data.atom.shape[0]
            target_dtype = CategoricalType
            target_labels = hfile.root.categorical_data.attrs.columns
            target_counts = list(hfile.root.categorical_counts)
            target_map = list(hfile.root.categorical_mappings)
        elif hasattr(hfile.root, "ordinal_data"):
            target_counts = None
            target_map = None
            target_labels = hfile.root.ordinal_data.attrs.columns
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
                         target_counts=target_counts)
    return m

def write_metadata(directory, m):
    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(m, f)


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

