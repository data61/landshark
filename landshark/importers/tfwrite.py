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
        writer.add(d)
    writer.close()


def training(data: Iterator[TrainingBatch], output_directory: str,
             test_frac: float, random_seed: int=666) -> int:
    test_directory = os.path.join(output_directory, "testing")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    writer = _MultiFileWriter(output_directory, tag="train")
    test_writer= _MultiFileWriter(test_directory, tag="test")
    rnd = np.random.RandomState(random_seed)

    n_train = 0
    for d in data:
        train_batch, test_batch = _split_on_mask(d, rnd, test_frac)
        n_train += train_batch.y.shape[0]
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
    if hasattr(batch, 'y'):
        it = zip(batch.x_ord, batch.x_cat, batch.y)
    else:
        it = zip(batch.x_ord, batch.x_cat, repeat(np.zeros(3)))
    for x_ord, x_cat, y in it:
        fdict = _make_features(x_ord, x_cat, y)
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
    n = d.y.shape[0]
    mask = rnd.choice([True, False], size=(n,),
                      p=[test_frac, 1.0 - test_frac])
    nmask = ~mask
    test_batch = TrainingBatch(x_ord=d.x_ord[mask],
                               x_cat=d.x_cat[mask],
                               y=d.y[mask])
    train_batch = TrainingBatch(x_ord=d.x_ord[nmask],
                                x_cat=d.x_cat[nmask],
                                y=d.y[nmask])
    return train_batch, test_batch
