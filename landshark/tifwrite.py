"""TIF writing functionality."""
import os.path
import logging

from typing import List, Iterator, cast, Optional
import numpy as np
import rasterio as rs
from rasterio.windows import Window

from landshark.metadata import TrainingMetadata
from landshark.image import ImageSpec
from landshark.basetypes import CategoricalType, OrdinalType, NumericalType,\
    RegressionPrediction, ClassificationPrediction, Prediction


log = logging.getLogger(__name__)

class BatchWriter:

    def __init__(self, rs_file: rs.DatasetReader, width: int, height: int,
                 dtype: NumericalType) -> None:
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


def _make_writer(directory: str, label: str, dtype: NumericalType,
                 image_spec: ImageSpec) -> BatchWriter:
    crs = rs.crs.CRS(**image_spec.crs)
    params = dict(driver="GTiff", width=image_spec.width,
                  height=image_spec.height, count=1, dtype=dtype,
                  crs=crs, affine=image_spec.affine)
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


def write_geotiffs(y_dash: Iterator[Prediction],
                   directory: str,
                   metadata: TrainingMetadata,
                   percentiles: Optional[List[float]],
                   tag: str="") -> None:
    classification = metadata.target_dtype != OrdinalType

    if percentiles is None:
        percentiles = []

    log.info("Initialising Geotiff writers")
    log.info("Image width: {} height: {}".format(metadata.image_spec.width,
                                                 metadata.image_spec.height))
    labels = [l + "_" + tag for l in metadata.target_labels]

    if classification:
        y_dash_c = cast(Iterator[ClassificationPrediction], y_dash)
        _write_classification(y_dash_c, labels, directory, metadata)
    else:
        y_dash_r = cast(Iterator[RegressionPrediction], y_dash)
        _write_regression(y_dash_r, labels, directory, metadata, percentiles)


def _write_classification(y_dash: Iterator[ClassificationPrediction],
                          labels: List[str],
                          directory: str,
                          metadata: TrainingMetadata) -> None:
    assert len(labels) == 1
    if metadata.target_map is None:
        raise ValueError("Cant write classification target without"
                         "target mapping")
    label = labels[0]
    ey_writer = _make_writer(directory, label, CategoricalType,
                             metadata.image_spec)
    p_labels = _make_classify_labels(label, metadata.target_map)
    p_writers = [_make_writer(directory, l, OrdinalType,
                              metadata.image_spec) for l in p_labels]
    for b, yb in enumerate(y_dash):
        ey_writer.write(yb.Ey.flatten())
        if yb.probabilities is not None:
            for d, w in zip(yb.probabilities.T, p_writers):
                w.write(d)
    ey_writer.close()
    for w in p_writers:
        w.close()


def _write_regression(y_dash: Iterator[RegressionPrediction],
                      labels: List[str],
                      directory: str,
                      metadata: TrainingMetadata,
                      percentiles: List[float]) -> None:
        perc_labels = [[l + "_p{}".format(p) for l in labels]
                       for p in percentiles]

        m_writers = [_make_writer(directory, l, OrdinalType,
                                  metadata.image_spec)
                     for l in labels]
        p_writers = [[_make_writer(directory, lbl, OrdinalType,
                                   metadata.image_spec) for lbl in lbl_list]
                     for lbl_list in perc_labels]

        for i, yi in enumerate(y_dash):
            mbatch = yi.Ey
            pbatch = yi.percentiles
            # write mean data
            for ym, mwriter in zip(mbatch.T, m_writers):
                mwriter.write(ym)
            # write perc data
            if pbatch is not None:
                for perc, pwriterlist in zip(pbatch, p_writers):
                    for bandperc, pwriter in zip(perc.T, pwriterlist):
                        pwriter.write(bandperc)

        log.info("Closing file objects")
        for j in m_writers:
            j.close()
        for k in p_writers:
            for m in k:
                m.close()
