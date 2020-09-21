"""Utilities to support iteration."""

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

import itertools
from typing import Iterator, List, Tuple, TypeVar

import numpy as np

from landshark.basetypes import FixedSlice

T = TypeVar("T")


def batch(it: Iterator[T], batchsize: int) -> Iterator[List[T]]:
    """Group iterator into batches."""
    while True:
        batch = list(itertools.islice(it, batchsize))
        if not batch:
            return
        yield batch


def batch_slices(batchsize: int, total_size: int) -> Iterator[FixedSlice]:
    """Group range indices into slices of a given batchsize."""
    n = total_size // batchsize
    ret = [(i * batchsize, (i + 1) * batchsize) for i in range(n)]
    if total_size % batchsize != 0:
        ret.append((n * batchsize, total_size))

    for start, stop in ret:
        yield FixedSlice(start, stop)


def with_slices(it: Iterator[np.ndarray]) -> Iterator[Tuple[FixedSlice, np.ndarray]]:
    """Add slice into vstacked array to each sub array in `it`."""
    start_idx = 0
    for d in it:
        end_idx = start_idx + d.shape[0]
        yield FixedSlice(start_idx, end_idx), d
        start_idx = end_idx
