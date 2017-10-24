"""Export data to tensorflow formats."""

import os.path
from itertools import chain
from collections import namedtuple
import pickle

import numpy as np
import tensorflow as tf
from typing import Iterator, List

from landshark.feed import TrainingBatch


RecordShape = namedtuple("RecordShape", [
    "x_ord",
    "x_cat",
    "halfwidth",
    "N",
    "ncategories"
    ])


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


def to_tfrecords(data: Iterator[TrainingBatch], output_directory: str,
                 name: str, ncategories: List[int]) -> None:
    """Do stuff."""
    directory = os.path.join(output_directory, "tfrecords_" + name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, "00001.tfrecord")
    with tf.python_io.TFRecordWriter(path) as writer:

        N = 0

        d0 = next(data)
        for d in chain([d0], data):
            N += d.x_ord.shape[0]
            for x_ord, x_cat, y in zip(d.x_ord, d.x_cat, d.y):
                fdict = _make_features(x_ord, x_cat, y)
                example = tf.train.Example(
                    features=tf.train.Features(feature=fdict))

                writer.write(example.SerializeToString())

    shape = RecordShape(x_ord=d0.x_ord.shape[3],
                        x_cat=d0.x_cat.shape[3],
                        halfwidth=(d0.x_ord.shape[1] - 1) // 2,
                        N=N,
                        ncategories=ncategories)

    spec_path = os.path.join(directory, "METADATA.bin")
    with open(spec_path, "wb") as f:
        pickle.dump(shape, f)
