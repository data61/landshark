"""TIF writing functionality."""

import logging
import os.path
from typing import Dict, Iterator, List, Optional, cast

import numpy as np
import rasterio as rs
from rasterio.windows import Window

from landshark.basetypes import (CategoricalType, NumericalType, OrdinalType)
from landshark.image import ImageSpec
from landshark.metadata import (CategoricalMetadata, OrdinalMetadata,
                                TrainingMetadata)
from landshark.errors import PredictionShape

log = logging.getLogger(__name__)


class BatchWriter:

    def __init__(self, rs_file: rs.DatasetReader, width: int, height: int,
                 dtype: np.dtype) -> None:
        self.f = rs_file
        self.width = width
        self.height = height
        self.dtype = dtype
        self.res = np.array([], dtype=dtype)
        self.rows_written = 0

    def write(self, data: np.ndarray) -> None:

        assert data.ndim == 1
        all_data = np.hstack((self.res, data))
        nrows = len(all_data) // self.width
        if nrows > 0:
            d = all_data[0: nrows * self.width].reshape(nrows, self.width)
            w = Window(0, self.rows_written, d.shape[1], d.shape[0])
            self.f.write(d, 1, window=w)
            self.rows_written += nrows
            self.res = all_data[nrows * self.width:]
        else:
            self.res = all_data

    def close(self) -> None:
        self.f.close()


def _make_writer(directory: str, label: str, dtype: np.dtype,
                 image_spec: ImageSpec) -> BatchWriter:
    crs = rs.crs.CRS(**image_spec.crs)
    params = {
        "driver": "GTiff",
        "width": image_spec.width,
        "height": image_spec.height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": image_spec.affine
    }
    fname = os.path.join(directory, label + ".tif")
    f = rs.open(fname, 'w', **params)
    writer = BatchWriter(f, width=image_spec.width, height=image_spec.height,
                         dtype=dtype)
    return writer


def _make_classify_labels(label: str, target_map: np.ndarray) -> List[str]:
    target_list = target_map[0]

    # Binary
    if len(target_list) <= 2:
        labels = [label + "_p(y={})".format(target_list[1])]
        return labels

    # Multiclass
    labels = [label +
              "_{}_{}".format(i, s.decode() if isinstance(s, bytes) else s)
              for i, s in enumerate(target_list)]

    return labels


def write_geotiffs(y_dash: Iterator[Dict[str, np.ndarray]],
                   directory: str,
                   metadata: TrainingMetadata,
                   tag: str="") -> None:

    imspec = metadata.features.image
    log.info("Initialising Geotiff writers")
    log.info("Image width: {} height: {}".format(imspec.width,
                                                 imspec.height))

    # get the first prediction so we can see what we're dealing with
    y0 = next(y_dash)
    for k, v in y0.items():
        if not (v.ndim == 1 or (v.ndim == 2 and v.shape[1] == 1)):
            raise PredictionShape(k, v.shape)

    writers = {k: _make_writer(directory, k + "_" + tag, v.dtype,
                            metadata.features.image) for k, v in y0.items()}

    # write the initial data
    for k, v in y0.items():
        writers[k].write(v.flatten())

    for y_i in y_dash:
        for k, v in y_i.items():
            writers[k].write(v.flatten())

    for w in writers.values():
        w.close()
