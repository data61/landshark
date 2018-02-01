import os.path
import logging

import numpy as np
import rasterio as rs
from rasterio.windows import Window

from landshark.basetypes import CategoricalType, OrdinalType
# with ExitStack() as stack:
#     files = [stack.enter_context(open(fname)) for fname in filenames]

log = logging.getLogger(__name__)


class BatchWriter:

    def __init__(self, rs_file, width, height, dtype):
        self.f = rs_file
        self.width = width
        self.height = height
        self.dtype = dtype
        self.res = np.array([], dtype=dtype)
        self.rows_written = 0

    def write(self, data):

        assert data.ndim == 1
        all_data = np.hstack((self.res, data))
        nrows = len(all_data) // self.width
        if nrows > 0:
            # w = (slice(self.rows_written, self.rows_written + nrows),
                 # slice(0, self.width))
            d = all_data[0: nrows * self.width].reshape(nrows, self.width)
            w = Window(0, self.rows_written, d.shape[1], d.shape[0])
            self.f.write(d, 1, window=w)
            self.rows_written += nrows
            self.res = all_data[nrows * self.width:]
        else:
            self.res = all_data

    def close(self):
        self.f.close()


def _make_writer(directory, label, dtype, image_spec):
    crs = rs.crs.CRS(**eval(image_spec.crs))
    params = dict(driver="GTiff", width=image_spec.width,
                  height=image_spec.height, count=1, dtype=dtype,
                  crs=crs, affine=image_spec.affine)
    fname = os.path.join(directory, label + ".tif")
    f = rs.open(fname, 'w', **params)
    writer = BatchWriter(f, width=image_spec.width, height=image_spec.height,
                         dtype=dtype)
    return writer


def _make_classify_labels(label, target_map):
    target_list = target_map[0]
    labels = [label + "_{}_{}".format(i, s.decode())
              for i, s in enumerate(target_list)]
    return labels


def write_geotiffs(y_dash, directory, metadata, percentiles, tag=""):

    classification = metadata.target_dtype != OrdinalType

    if percentiles is None:
        percentiles = []

    log.info("Initialising Geotiff writers")
    log.info("Image width: {} height: {}".format(metadata.image_spec.width,
                                                 metadata.image_spec.height))
    labels = [l + "_" + tag for l in metadata.target_labels]

    if classification:
        _write_classification(y_dash, labels, directory, metadata)
    else:
        _write_regression(y_dash, labels, directory, metadata, percentiles)



def _write_classification(y_dash, labels, directory, metadata):
        assert len(labels) == 1
        label = labels[0]
        ey_writer = _make_writer(directory, label, CategoricalType,
                                 metadata.image_spec)
        p_labels = _make_classify_labels(label, metadata.target_map)
        p_writers = [_make_writer(directory, l, OrdinalType,
                                  metadata.image_spec) for l in p_labels]
        for b, (ey_batch, prob_batch) in enumerate(y_dash):
            ey_writer.write(ey_batch.flatten())
            for d, w in zip(prob_batch.T, p_writers):
                w.write(d)
        ey_writer.close()
        for w in p_writers:
            w.close()


def _write_regression(y_dash, labels, directory, metadata, percentiles):
        perc_labels = [[l + "_p{}".format(p) for l in labels]
                       for p in percentiles]

        m_writers = [_make_writer(directory, l, OrdinalType, metadata.image_spec)
                     for l in labels]
        p_writers = [[_make_writer(directory, lbl, OrdinalType,
                                   metadata.image_spec) for lbl in lbl_list]
                     for lbl_list in perc_labels]

        for i, (mbatch, pbatch) in enumerate(y_dash):
            # write mean data
            for ym, mwriter in zip(mbatch.T, m_writers):
                mwriter.write(ym)
            # write perc data
            for perc, pwriterlist in zip(pbatch, p_writers):
                for bandperc, pwriter in zip(perc.T, pwriterlist):
                    pwriter.write(bandperc)

        log.info("Closing file objects")
        for i in m_writers:
            i.close()
        for i in p_writers:
            for j in i:
                j.close()
