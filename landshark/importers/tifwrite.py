import os.path
import logging

import numpy as np
import rasterio as rs
from rasterio.windows import Window

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
        all_data = np.hstack((self.res, data))
        nrows = len(all_data) // self.width
        if nrows > 0:
            # w = (slice(self.rows_written, self.rows_written + nrows),
                 # slice(0, self.width))
            d = all_data[0: nrows*self.width].reshape(nrows, self.width)
            w = Window(0, self.rows_written, d.shape[1], d.shape[0])
            self.f.write(d, 1, window=w)
            self.rows_written += nrows
            self.res = all_data[nrows * self.width:]
        else:
            self.res = all_data

    def close(self):
        self.f.close()

def _make_writer(directory, label, metadata):
    dtype = rs.float32 if metadata.target_dtype == np.float32 else rs.int32
    image_spec = metadata.image_spec
    log.info("Image width: {} height: {}".format(image_spec.width,
                                                 image_spec.height))
    params = dict(driver="GTiff", width=image_spec.width,
                  height=image_spec.height, count=1, dtype=dtype,
                  crs=image_spec.crs, affine=image_spec.affine)
    fname = os.path.join(directory, label + ".tif")
    f = rs.open(fname, 'w', **params)
    writer = BatchWriter(f, width=image_spec.width, height=image_spec.height,
                         dtype=dtype)
    return writer


def write_geotiffs(y_dash, directory, metadata, percentiles, tag=""):
    log.info("Initialising Geotiff writer")
    labels = [l + "_" + tag for l in metadata.target_labels]
    perc_labels = [[l + "_p{}".format(p) for l in labels] for p in percentiles]

    m_writers = [_make_writer(directory, l, metadata)
                 for l in labels]
    p_writers = [[_make_writer(directory, lbl, metadata) for lbl in lbl_list]
                 for lbl_list in perc_labels]

    for i, (mbatch, pbatch) in enumerate(y_dash):
        log.info("Writing batch {} to disk".format(i))

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
