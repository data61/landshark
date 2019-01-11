"""Useful functions for simplifying model (config) files."""

from typing import Dict, Union

import numpy as np
import tensorflow as tf


def flatten_patch(x: tf.Tensor) -> tf.Tensor:
    """
    Reshape tensor by concatenating patch pixel columns.

    Arguments
    ---------
    x : tf.Tensor
        A tensor of shape (n,p,p,d) where the middle p is the patch width.

    Returns
    -------
    x_new : tf.Tensor
        A reshaped version of x, of shape (n, p * p * d).

    """
    new_shp = (tf.shape(x)[0], np.product(x.shape[1:]))
    new_x = tf.reshape(x, new_shp)
    return new_x

def value_impute(data: tf.Tensor, mask: tf.Tensor,
                 newval: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    """
    Impute missing (masked) values with a single value.
    """
    tmask = tf.cast(mask, dtype=data.dtype)
    fmask = tf.cast(tf.logical_not(mask), dtype=data.dtype)
    newdata = data * fmask + newval * tmask
    return newdata

def continuous_input(d: Dict[str, tf.Tensor]) -> tf.Tensor:
    cols = [tf.feature_column.numeric_column(k) for k in d.keys()]
    inputs = tf.feature_column.input_layer(d, cols)
    return inputs

def categorical_embedded_input(d: Dict[str, tf.Tensor],
                               ncat_dict: Dict[str, int],
                               embed_dict: Dict[str, int]) -> tf.Tensor:
    columns_cat = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
        key=k, num_buckets=(v + 1)), embed_dict[k])
        for k, v in ncat_dict.items()]
    inputs_cat = tf.feature_column.input_layer(d, columns_cat)
    return inputs_cat
