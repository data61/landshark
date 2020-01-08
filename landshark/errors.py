"""Exceptions and errors from CLI and internal processing."""

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

import logging
import sys
from typing import Any, Callable, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exceptions in Landshark."""

    def __init__(self, msg: str = "Landshark error."):
        super().__init__(msg)


def catch_and_exit(f: Callable) -> Callable:
    """Decorate function to exit program if it throws an Error."""

    def wrapped(*args: Any, **kwargs: Any) -> None:
        try:
            f(*args, **kwargs)
        except Error as e:
            log.error(f"{type(e).__name__}: {e}")
            sys.exit()

    return wrapped


class NoTifFilesFound(Error):
    """Couldn't find TIF files."""

    def __init__(self):
        super().__init__(
            "The supplied paths had no files with .tif or .gtif extensions"
        )


class ZeroDeviation(Error):
    """Zero standard deviation in supplied column."""

    def __init__(self, sd: np.ndarray, cols: List[str]) -> None:
        zsrcs = [c for z, c in zip(sd, cols) if z]
        super().__init__(
            f"The following sources have bands with zero standard deviation: {zsrcs}"
        )


class ConCatNMismatch(Error):
    """N doesnt match between the con and cat sources."""

    def __init__(self, N_con: int, N_cat: int) -> None:
        super().__init__(
            "Continuous and Categorical source mismatch with"
            f"{N_con} and {N_cat} points respectively"
        )


class InvalidPredictionShape(Error):
    """Prediction output is not 1D or 2D."""

    def __init__(self, name: str, shape: Tuple[int]) -> None:
        super().__init__(
            f"'{name}' prediction has shape {shape}. Predictions must be 1D."
        )
