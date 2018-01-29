from itertools import repeat

import tensorflow as tf
import numpy as np


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

def serialise(x_ord, x_cat, y):
    if x_ord is None:
        x_ord = repeat(np.ma.MaskedArray(data=[], mask=[]))
    if x_cat is None:
        x_cat = repeat(np.ma.MaskedArray(data=[], mask=[]))
    if y is None:
        # TODO dont know the dtype so this is a bit dodgy
        y = repeat(np.array([], dtype=np.float32))

    string_list = []
    for xo_i, xc_i, y_i in zip(x_ord, x_cat, y):
        fdict = _make_features(xo_i, xc_i, y_i)
        example = tf.train.Example(
            features=tf.train.Features(feature=fdict))
        string_list.append(example.SerializeToString())
    return string_list

