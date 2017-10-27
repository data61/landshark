
"""Metadata."""

import os.path
import pickle
from collections import namedtuple


TrainingMetadata = namedtuple("TrainingMetadata", [
    "nfeatures_ord",
    "nfeatures_cat",
    "halfwidth",
    "N",
    "ncategories"
    ])


def write_metadata(directory, npoints, nfeatures_cat, nfeatures_ord, halfwidth,
                   ncategories):
    shape = TrainingMetadata(nfeatures_ord=nfeatures_ord,
                             nfeatures_cat=nfeatures_cat,
                             halfwidth=halfwidth,
                             N=npoints,
                             ncategories=ncategories)

    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(shape, f)
