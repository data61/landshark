"""Export data to tensorflow formats."""

import logging
import os.path

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

FILESIZE_MB = 100


def query(data, n_total, output_directory, tag):
    writer = _MultiFileWriter(output_directory, tag=tag)
    for d in data:
        writer.add(d)
    writer.close()


def training(data, n_total: int, output_directory: str,
             testfold: int, folds: int, random_seed: int=666) -> int:
    test_directory = os.path.join(output_directory, "testing")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    writer = _MultiFileWriter(output_directory, tag="train")
    test_writer = _MultiFileWriter(test_directory, tag="test")
    rnd = np.random.RandomState(random_seed)

    n_train = 0
    for d in data:
        train_batch, test_batch = _split_on_mask(d, rnd, testfold, folds)
        n_train += len(train_batch)
        writer.add(train_batch)
        test_writer.add(test_batch)
    writer.close()
    test_writer.close()
    return n_train


def _get_mb(path):
    filesize = os.path.getsize(path) // (1024 ** 2)
    return filesize


class _MultiFileWriter:
    def __init__(self, output_directory, tag):
        self.output_directory = output_directory
        self.tag = tag
        self.file_index = -1
        self._options = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.ZLIB)
        self._nextfile()
        self.lines_written = 0

    def _nextfile(self):
        if hasattr(self, '_f'):
            self._f.close()
        self.file_index += 1
        self.path = os.path.join(
            self.output_directory,
            "{}.{:05d}.tfrecord".format(self.tag, self.file_index))
        self._f = tf.python_io.TFRecordWriter(self.path, options=self._options)

    def add(self, batch):
        filesize = _get_mb(self.path)
        if filesize > FILESIZE_MB:
            self._nextfile()
        for b in batch:
            self._f.write(b)
            self.lines_written += 1
        self._f.flush()

    def close(self):
        self._f.close()


def _split_on_mask(data, rnd, testfold, folds):
    n = len(data)
    mask = rnd.randint(1, folds + 1, size=(n,)) != testfold
    nmask = ~mask
    train_batch = [data[i] for i, m in enumerate(mask) if m]
    test_batch = [data[i] for i, m in enumerate(nmask) if m]
    return train_batch, test_batch
