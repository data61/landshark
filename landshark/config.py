"""Useful functions for simplifying model (config) files."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Union

import numpy as np
import tensorflow as tf


@tf.function
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
    new_shp = (tf.shape(input=x)[0], np.product(x.shape[1:]))
    new_x = tf.reshape(x, new_shp)
    return new_x


@tf.function
def value_impute(
    data: tf.Tensor, mask: tf.Tensor, newval: Union[tf.Tensor, np.ndarray]
) -> tf.Tensor:
    """Impute missing (masked) values with a single value."""
    tmask = tf.cast(mask, dtype=data.dtype)
    fmask = tf.cast(tf.logical_not(mask), dtype=data.dtype)
    newdata = data * fmask + newval * tmask
    return newdata


def continuous_input(d: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Create input layer for named continuous data."""
    cols = [tf.feature_column.numeric_column(k) for k in d.keys()]
    inputs = tf.keras.layers.DenseFeatures(cols)(d)
    return inputs


def categorical_embedded_input(
    d: Dict[str, tf.Tensor], ncat_dict: Dict[str, int], embed_dict: Dict[str, int]
) -> tf.Tensor:
    """Create input layer for named categorical data with embedding."""
    columns_cat = [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                key=k, num_buckets=(v + 1)
            ),
            embed_dict[k],
        )
        for k, v in ncat_dict.items()
    ]

    inputs_cat = tf.keras.layers.DenseFeatures(columns_cat)(d)
    return inputs_cat
