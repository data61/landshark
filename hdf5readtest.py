from time import time, sleep
import numpy as np

import tables

MAXROWS = 500000
# MAXROWS = np.inf
BATCH = 1000
READS = 10
# tables.parameters.CHUNK_CACHE_SIZE = 1000 * 1048576
# tables.parameters.CHUNK_CACHE_PREEMPT = 0.5


def timethis(func):
    def wrapper(*args, **kwargs):
        start = time()
        ret = func(*args, **kwargs)
        stop = time()
        print("{} ran for {:0.4}s".format(func.__name__, stop - start))
        return ret

    return wrapper


@timethis
def read_whole():
    hfile = tables.open_file("lbalpha.hdf5")
    print("Chunk shape {}".format(hfile.root.ordinal_data.chunkshape))
    ord_data = hfile.root.ordinal_data.read()

    hfile.close()


@timethis
def read_rows():
    hfile = tables.open_file("lbalpha.hdf5")
    ord_data = hfile.root.ordinal_data
    rows = min(MAXROWS, ord_data.shape[0])
    container = np.zeros(ord_data.shape[1], dtype=np.float32)

    # list(map(lambda r: ord_data.read(r, out=container), range(rows)))
    # print(container)
    for r in range(rows):
        container[:] = ord_data[r, :]

    hfile.close()


@timethis
def read_rows_into():
    hfile = tables.open_file("lbalpha.hdf5")
    ord_data = hfile.root.ordinal_data
    rows = min(MAXROWS, ord_data.shape[0])
    container = np.empty(ord_data.shape[1], dtype=np.float32)

    for r in range(rows):
        ord_data.read(r, out=container)

    hfile.close()


@timethis
def read_row_batches():
    hfile = tables.open_file("lbalpha.hdf5")
    ord_data = hfile.root.ordinal_data
    last_row = min(MAXROWS, ord_data.shape[0])
    rows_b = range(0, last_row - BATCH, BATCH)
    rows_e = range(BATCH, last_row, BATCH)
    container = np.empty((BATCH, ord_data.shape[1]))

    for b, e in zip(rows_b, rows_e):
        container[:] = ord_data[b:e, :]

    hfile.close()


@timethis
def read_row_batches_offset():
    hfile = tables.open_file("lbalpha.hdf5")
    ord_data = hfile.root.ordinal_data
    batch = BATCH // READS
    last_row = min(MAXROWS, ord_data.shape[0]) // READS
    rows_b = range(0, last_row - batch, batch)
    rows_e = range(batch, last_row, batch)
    container = np.empty((batch, ord_data.shape[1]))

    for b, e in zip(rows_b, rows_e):
        for i in range(READS):
            container[:] = ord_data[(b + i * last_row):(e + i * last_row), :]

    hfile.close()


@timethis
def read_cols():
    hfile = tables.open_file("lbalpha.hdf5")
    ord_data = hfile.root.ordinal_data
    cols = ord_data.shape[1]
    last_row = min(MAXROWS, ord_data.shape[0])
    container = np.empty(last_row)

    for c in range(cols):
        container[:] = ord_data[:last_row, c]

    hfile.close()


if __name__ == "__main__":
    # read_whole()
    # read_row_batches()
    # read_row_batches_offset()
    # read_rows_into()
    read_rows()
    # read_cols()
