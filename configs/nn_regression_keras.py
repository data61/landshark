"""Example regression config using a Keras Model."""

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

from typing import List, Tuple

import tensorflow as tf

from landshark.kerasmodel import FeatInput, get_feat_input_list, impute_const_layer


def r2(y_true, y_pred):
    """Coefficient of determination metric."""
    SS_res = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))
    SS_tot = tf.reduce_sum(tf.math.squared_difference(y_true, tf.reduce_mean(y_true)))
    return 1 - SS_res / SS_tot


def model(
    num_feats: List[FeatInput],
    cat_feats: List[Tuple[FeatInput, int]],
    indices: tf.keras.Input,
    coords: tf.keras.Input,
    target_labels: List[str],
) -> tf.keras.Model:
    """ Example model config.
        Must match the signature above and return a compiled tf.keras.Model
    """

    # inpute with constant value
    num_imputed = [impute_const_layer(x) for x in num_feats]
    cat_imputed = [impute_const_layer(x, n) for x, n in cat_feats]

    # embed categorical
    embed_dims = [(n, 3) for _, n in cat_feats]
    cat_embedded = [
        tf.keras.layers.Embedding(n, d)(tf.squeeze(x, 3))
        for x, (n, d) in zip(cat_imputed, embed_dims)
    ]

    # CNN on patch data
    l0 = tf.keras.layers.Concatenate(axis=3)(num_imputed + cat_embedded)
    l1 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation="relu")(l0)
    l2 = tf.keras.layers.Conv2D(filters=18, kernel_size=2, activation="relu")(l1)

    # Get some predictions for the labels
    n_targets = len(target_labels)
    l3 = tf.keras.layers.Dense(units=n_targets, activation=None)(l2)
    phi = [tf.reshape(l3[..., i], (-1, 1), name=l) for i, l in enumerate(target_labels)]

    # create keras model
    model_inputs = get_feat_input_list(num_feats, cat_feats)
    model = tf.keras.Model(inputs=model_inputs, outputs=phi)
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=[r2])

    return model
