"""High-level file IO operations and utility functions."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
from glob import glob
from typing import List


def tifnames(directories: List[str]) -> List[str]:
    """Recursively find all tif/gtif files within a list of directories."""
    names: List[str] = []
    for d in directories:
        file_types = ("tif", "gtif")
        for t in file_types:
            if os.path.isfile(d) and d.endswith(f".{t}"):
                names.append(d)
                break

            glob_pattern = os.path.join(d, "**", "*.{}".format(t))
            names.extend(glob(glob_pattern, recursive=True))

    return names
