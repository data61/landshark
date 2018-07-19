"""High-level file IO operations and utility functions."""

import os.path
from glob import glob
from typing import List


def tifnames(directories: List[str]) -> List[str]:
    names: List[str] = []
    for d in directories:
        file_types = ("tif", "gtif")
        for t in file_types:
            glob_pattern = os.path.join(d, "**", "*.{}".format(t))
            names.extend(glob(glob_pattern, recursive=True))
    return names


def parse_withlist(listfile: str) -> List[str]:
    with open(listfile, "r") as f:
        lines = f.readlines()
    # remove the comment lines
    nocomments = [l.split("#")[0] for l in lines]
    stripped = [l.strip().rstrip() for l in nocomments]
    noempty = [l for l in stripped if l is not ""]
    return noempty
