
"""Metadata."""

import os.path
import pickle
from collections import namedtuple


TrainingMetadata = namedtuple("TrainingMetadata", [
    "ntargets",
    "target_dtype",
    "nfeatures_ord",
    "nfeatures_cat",
    "halfwidth",
    "N",
    "ncategories"
    ])


def write_metadata(directory, npoints, ntargets, target_dtype, nfeatures_cat,
                   nfeatures_ord, halfwidth, ncategories):
    m = TrainingMetadata(ntargets=ntargets,
                         target_dtype=target_dtype,
                         nfeatures_ord=nfeatures_ord,
                         nfeatures_cat=nfeatures_cat,
                         halfwidth=halfwidth,
                         N=npoints,
                         ncategories=ncategories)

    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(m, f)
