"""Export data to tensorflow formats."""

import logging
import os.path
from itertools import chain, count, repeat

import numpy as np
import tensorflow as tf
from typing import Iterator

from landshark.feed import TrainingBatch

log = logging.getLogger(__name__)


FILESIZE = 100

def query(data, output_directory, tag):
    writer = _MultiFileWriter(output_directory, tag=tag)
    for d in data:
        writer.add((d[0], d[1], None))
    writer.close()


def training(data, output_directory: str,
             test_frac: float, random_seed: int=666) -> int:
    test_directory = os.path.join(output_directory, "testing")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    writer = _MultiFileWriter(output_directory, tag="train")
    test_writer = _MultiFileWriter(test_directory, tag="test")
    rnd = np.random.RandomState(random_seed)

    n_train = 0
    for d in data:
        train_batch, test_batch = _split_on_mask(d, rnd, test_frac)
        n_train += train_batch[2].shape[0]
        writer.add(train_batch)
        test_writer.add(test_batch)
    writer.close()
    test_writer.close()
    return n_train


def _ndarray_feature(x: np.ndarray) -> tf.train.Feature:
    """Create an ndarray feature stored as bytes."""
    x_bytes = x.flatten().tostring()
    feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[x_bytes]))
    return feature


def _make_features(x_ord: np.ma.MaskedArray, x_cat: np.ma.MaskedArray,
                   y: np.ndarray) -> dict:
    """Do stuff."""
    fdict = {
        "x_cat": _ndarray_feature(x_cat.data),
        "x_cat_mask": _ndarray_feature(x_cat.mask),
        "x_ord": _ndarray_feature(x_ord.data),
        "x_ord_mask": _ndarray_feature(x_ord.mask),
        "y": _ndarray_feature(y)
        }
    return fdict


def _write(batch, writer):

    x_ord, x_cat, y = batch

    if x_ord is None:
        x_ord = repeat(np.ma.MaskedArray(data=[], mask=[]))
    if x_cat is None:
        x_cat = repeat(np.ma.MaskedArray(data=[], mask=[]))
    if y is None:
        # TODO dont know the dtype so this is a bit dodgy
        y = repeat(np.array([], dtype=np.float32))

    for xo_i, xc_i, y_i in zip(x_ord, x_cat, y):
        fdict = _make_features(xo_i, xc_i, y_i)
        example = tf.train.Example(
            features=tf.train.Features(feature=fdict))
        writer.write(example.SerializeToString())


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
        if filesize > FILESIZE:
            self._nextfile()
        _write(batch, self._f)

    def close(self):
        self._f.close()


def _split_on_mask(d, rnd, test_frac):
    n = d[2].shape[0]
    mask = rnd.choice([True, False], size=(n,),
                      p=[test_frac, 1.0 - test_frac])
    nmask = ~mask

    train_batch = [None] * 3
    test_batch = [None] * 3

    for i, d_i in enumerate(d):
        if d_i is not None:
            train_batch[i] = d_i[nmask]
            test_batch[i] = d_i[mask]
    train_batch = tuple(train_batch)
    test_batch = tuple(test_batch)

    return train_batch, test_batch
