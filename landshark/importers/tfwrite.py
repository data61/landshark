"""Export data to tensorflow formats."""

import os.path
from itertools import chain
from collections import namedtuple
import pickle

import numpy as np
import tensorflow as tf
from typing import Iterator, List

from landshark.feed import TrainingBatch


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


def _write_batch(batch, writer):
    for x_ord, x_cat, y in zip(batch.x_ord, batch.x_cat, batch.y):
        fdict = _make_features(x_ord, x_cat, y)
        example = tf.train.Example(
            features=tf.train.Features(feature=fdict))
        writer.write(example.SerializeToString())


def to_tfrecords(data: Iterator[TrainingBatch], output_directory: str,
                 test_frac: float, random_seed: int=666) -> None:
    """Do stuff."""
    test_directory = os.path.join(output_directory, "testing")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    path = os.path.join(output_directory, "00001.tfrecord")
    test_path = os.path.join(test_directory, "00001.tfrecord")

    rnd = np.random.RandomState(random_seed)

    with tf.python_io.TFRecordWriter(path) as writer:
        with tf.python_io.TFRecordWriter(test_path) as test_writer:

            d0 = next(data)
            for d in chain([d0], data):
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
                _write_batch(train_batch, writer)
                _write_batch(test_batch, test_writer)
