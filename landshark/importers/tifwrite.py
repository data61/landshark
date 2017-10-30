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
        all_data = np.concatenate((self.res, data), axis=0)
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

def _make_writer(directory, label, metadata, image_spec):
    dtype = rs.float32 if metadata.target_dtype == np.float32 else rs.int32
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


def write_geotiffs(y_dash, directory, metadata, image_spec):

    log.info("Initialising Geotiff writer")
    labels = metadata.target_labels
    std_labels = [l + "_std" for l in metadata.target_labels]

    writers = [_make_writer(directory, l, metadata, image_spec)
               for l in labels]
    std_writers = [_make_writer(directory, l, metadata, image_spec)
                   for l in std_labels]

    for i, (ys, ys_std) in enumerate(y_dash):
        log.info("Writing batch {} to disk".format(i))
        for ys_i, writer in zip(ys.T, writers):
            writer.write(ys_i)
        for yss_i, swriter in zip(ys_std.T, std_writers):
            swriter.write(yss_i)

    log.info("Closing file objects")
    for w in writers:
        w.close()
    for w in std_writers:
        w.close()
