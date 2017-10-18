"""Export data to tensorflow formats."""

import os.path

import numpy as np
import tensorflow as tf
from typing import Iterator

from landshark.feed import TrainingBatch


def _ndarray_feature(x: np.ndarray) -> tf.train.Feature:
    """Create an ndarray feature stored as bytes."""
    x_bytes = x.flatten().tostring()
    feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[x_bytes]))
    return feature


def _int64_vec_feature(x: np.ndarray) -> tf.train.Feature:
    """Create a int64 vector feature."""
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=x))
    return feature


def _make_features(x_ord: np.ma.MaskedArray, x_cat: np.ma.MaskedArray,
                   y: np.ndarray) -> dict:
    """Do stuff."""
    fdict = {
        "x_cat": _ndarray_feature(x_cat.data),
        "x_cat_mask": _ndarray_feature(x_cat.mask),
        "x_cat_shape": _int64_vec_feature(x_cat.shape),
        "x_ord": _ndarray_feature(x_ord.data),
        "x_ord_mask": _ndarray_feature(x_ord.mask),
        "x_ord_shape": _int64_vec_feature(x_ord.shape),
        "y": _ndarray_feature(y)
        }
    return fdict


def to_tfrecords(data: Iterator[TrainingBatch],
                 output_directory: str, name: str) -> None:
    """Do stuff."""
    directory = os.path.join(output_directory, "tfrecords_" + name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, "00001.tfrecord")

    with tf.python_io.TFRecordWriter(path) as writer:
        for d in data:
            for x_ord, x_cat, y in zip(d.x_ord, d.x_cat, d.y):
                fdict = _make_features(x_ord, x_cat, y)
                example = tf.train.Example(
                    features=tf.train.Features(feature=fdict))

                writer.write(example.SerializeToString())

    # import IPython; IPython.embed(); import sys; sys.exit()
