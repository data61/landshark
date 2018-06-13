"""High-level file IO operations and utility functions."""

import os.path
from glob import glob

from typing import Optional, List


def tifnames(directories: List[str]) -> List[str]:
    names: List[str] = []
    for d in directories:
        file_types = ("tif", "gtif")
        for t in file_types:
            glob_pattern = os.path.join(d, "**", "*.{}".format(t))
            names.extend(glob(glob_pattern, recursive=True))
    return names

