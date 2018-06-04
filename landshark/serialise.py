"""Serialise and Deserialise to and from tf records."""
from itertools import repeat
from typing import Tuple, List

import tensorflow as tf
import numpy as np

from landshark.basetypes import OrdinalType
from landshark.metadata import TrainingMetadata, CategoricalMetadata


#
# Module constants and types
#

_FDICT = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }


#
# Module functions
#

def serialise(x_ord: np.ma.MaskedArray, x_cat: np.ma.MaskedArray,
              y: np.array) -> List[bytes]:
    """Serialise data to tf.records."""
    if x_ord is None:
        x_ord = repeat(np.ma.MaskedArray(data=[], mask=[]))
    if x_cat is None:
        x_cat = repeat(np.ma.MaskedArray(data=[], mask=[]))
    if y is None:
        # TODO dont know the dtype so this is a bit dodgy
        y = repeat(np.array([], dtype=OrdinalType))

    print("serialising {}, {}".format(x_ord.shape, x_cat.shape))

    string_list = []
    for xo_i, xc_i, y_i in zip(x_ord, x_cat, y):
        fdict = _make_features(xo_i, xc_i, y_i)
        example = tf.train.Example(
            features=tf.train.Features(feature=fdict))
        string_list.append(example.SerializeToString())
    return string_list


def deserialise(iterator: tf.data.Iterator, metadata: TrainingMetadata) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decode tf.record strings into Tensors."""
    str_features = iterator.get_next()
    raw_features = tf.parse_example(str_features, features=_FDICT)
    npatch = (2 * metadata.halfwidth + 1) ** 2
    categorical = isinstance(metadata.targets, CategoricalMetadata)
    y_type = tf.int32 if categorical else tf.float32
    with tf.name_scope("Inputs"):
        x_ord = tf.decode_raw(raw_features["x_ord"], tf.float32)
        x_cat = tf.decode_raw(raw_features["x_cat"], tf.int32)
        x_ord_mask = tf.decode_raw(raw_features["x_ord_mask"], tf.uint8)
        x_cat_mask = tf.decode_raw(raw_features["x_cat_mask"], tf.uint8)
        x_ord_mask = tf.cast(x_ord_mask, tf.bool)
        x_cat_mask = tf.cast(x_cat_mask, tf.bool)
        y = tf.decode_raw(raw_features["y"], y_type)

        nfeatures_ord = metadata.features.ordinal.D \
            if metadata.features.ordinal else 0
        nfeatures_cat = metadata.features.categorical.D \
            if metadata.features.categorical else 0
        ntargets = metadata.targets.D

        x_ord.set_shape((None, npatch * nfeatures_ord))
        x_ord_mask.set_shape((None, npatch * nfeatures_ord))
        x_cat.set_shape((None, npatch * nfeatures_cat))
        x_cat_mask.set_shape((None, npatch * nfeatures_cat))
        y.set_shape((None, ntargets))

    return x_ord, x_ord_mask, x_cat, x_cat_mask, y


#
# Private module utilities
#

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
