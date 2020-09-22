"""Utilities."""

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
from typing import Tuple

import numpy as np

from landshark.basetypes import (
    CategoricalType,
    ContinuousType,
    CoordinateType,
    MissingType,
)
from landshark.metadata import FeatureSet

log = logging.getLogger(__name__)


def to_masked(array: np.ndarray, missing_value: MissingType) -> np.ma.MaskedArray:
    """Create a masked array from array plus list of missing."""
    if missing_value is None:
        marray = np.ma.MaskedArray(data=array, mask=np.ma.nomask)
    else:
        mask = array == missing_value
        marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


def _batch_points(
    batchMB: float,
    ndim_con: int,
    ndim_cat: int,
    ndim_coord: int = 0,
    halfwidth: int = 0,
) -> Tuple[float, float]:
    patchsize = (halfwidth * 2 + 1) ** 2
    bytes_con = np.dtype(ContinuousType).itemsize * ndim_con
    bytes_cat = np.dtype(CategoricalType).itemsize * ndim_cat
    bytes_coord = np.dtype(CoordinateType).itemsize * ndim_coord
    mbytes_per_point = (bytes_con + bytes_cat + bytes_coord) * patchsize * 1e-6
    npoints = batchMB / mbytes_per_point
    return npoints, mbytes_per_point


def mb_to_points(
    batchMB: float,
    ndim_con: int,
    ndim_cat: int,
    ndim_coord: int = 0,
    halfwidth: int = 0,
) -> int:
    """Calculate the number of points of data to fill a memory allocation."""
    log.info("Batch size of {}MB requested".format(batchMB))
    npoints, mb_per_point = _batch_points(
        batchMB, ndim_con, ndim_cat, ndim_coord, halfwidth
    )
    npoints = int(round(max(1.0, npoints)))
    log.info(
        "Batch size set to {} points, total {:0.2f}MB".format(
            npoints, npoints * mb_per_point
        )
    )
    return npoints


def mb_to_rows(batchMB: float, row_width: int, ndim_con: int, ndim_cat: int) -> int:
    """Calculate the number of rows of data to fill a memory allocation."""
    log.info("Batch size of {}MB requested".format(batchMB))
    npoints, mb_per_point = _batch_points(batchMB, ndim_con, ndim_cat)
    nrows = int(round(max(1.0, npoints / row_width)))
    log.info(
        "Batch size set to {} rows, total {:0.2f}MB".format(
            nrows, mb_per_point * row_width * nrows
        )
    )
    return nrows


def points_per_batch(meta: FeatureSet, batch_mb: float) -> int:
    """Calculate batchsize in points given a memory allocation."""
    ndim_con = len(meta.continuous.columns) if meta.continuous else 0
    ndim_cat = len(meta.categorical.columns) if meta.categorical else 0
    batchsize = mb_to_points(batch_mb, ndim_con, ndim_cat, halfwidth=meta.halfwidth)
    return batchsize
