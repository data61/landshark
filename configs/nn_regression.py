"""Generic classification config file."""

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

from typing import Dict, Optional

import tensorflow as tf
from tensorflow import estimator

from landshark import config as utils
from landshark.metadata import Training


def model(mode: estimator.ModeKeys,
          X_con: Optional[Dict[str, tf.Tensor]],
          X_con_mask: Optional[Dict[str, tf.Tensor]],
          X_cat: Optional[Dict[str, tf.Tensor]],
          X_cat_mask: Optional[Dict[str, tf.Tensor]],
          Y: tf.Tensor,
          image_indices: tf.Tensor,
          coordinates: tf.Tensor,
          metadata: Training,
          ) -> tf.estimator.EstimatorSpec:
    """
    Describe the specification of a Tensorflow custom estimator model.

    This function must be implemented in all configurations. It is almost
    exactly the model function passed to a custom Tensorflow estimator,
    apart from having more convenient input arguments.
    See https://www.tensorflow.org/guide/custom_estimators

    Parameters
    ----------
    mode : tf.estimator.ModeKeys
        One of TRAIN, TEST or EVAL, describing in which context this code
        is being run by the estimator.
    coordinates : tf.Tensor
        A (?, 2) the world coordinates (x, y) of features
    X_con : dict
        A dictionary of tensors containing the continuous feature columns.
        Indexed by column names.
    X_con_mask : dict
        A dictionary of tensors containing the masks for the continuous
        feature columns.
    X_cat : dict
        A dictionary of tensors containing the categorical feature columns.
        Indexed by column names.
    X_cat_mask : dict
        A dictionary of tensors containing the masks for the categorical
        feature columns.
    Y : tf.Tensor
        A (?, k) tensor giving the k targets for the prediction.
    image_indices : tf.Tensor
        A (?, 2) the image coordinates of features
    params : dict
        Extra params given by the estimator.
    metadata : Metadata
        The metadata object has comprehensive information about the features
        and targets useful for model building (for example, the number of
        possible values for each categorical column). For more details
        check the landshark documentation.

    Returns
    -------
    tf.estimator.EstimatorSpec
        An EstimatorSpec object describing the model. For details check
        the Tensorflow custom estimator howto.

    """
    inputs_list = []
    if X_con:
        assert X_con_mask
        # let's 0-impute continuous columns
        X_con = {k: utils.value_impute(X_con[k], X_con_mask[k],
                                       tf.constant(0.0)) for k in X_con}

        # just concatenate the patch pixels as more features
        X_con = {k: utils.flatten_patch(v) for k, v in X_con.items()}

        # convenience function for catting all columns into tensor
        inputs_con = utils.continuous_input(X_con)
        inputs_list.append(inputs_con)

    if X_cat:
        assert X_cat_mask and metadata.features.categorical
        # impute as an extra category
        extra_cat = {
            k: metadata.features.categorical.columns[k].mapping.shape[0]
            for k in X_cat
        }
        X_cat = {k: utils.value_impute(
            X_cat[k], X_cat_mask[k], tf.constant(extra_cat[k]))
            for k in X_cat}
        X_cat = {k: utils.flatten_patch(v) for k, v in X_cat.items()}

        nvalues = {k: v.nvalues + 1 for k, v in
                   metadata.features.categorical.columns.items()}
        embedding_dims = {k: 3 for k in X_cat.keys()}
        inputs_cat = utils.categorical_embedded_input(X_cat, nvalues,
                                                      embedding_dims)
        inputs_list.append(inputs_cat)

    # Build a simple 2-layer network
    inputs = tf.concat(inputs_list, axis=1)
    l1 = tf.compat.v1.layers.dense(inputs, units=64, activation=tf.nn.relu)
    l2 = tf.compat.v1.layers.dense(l1, units=32, activation=tf.nn.relu)

    # Get some predictions for the labels
    phi = tf.compat.v1.layers.dense(l2, units=metadata.targets.D, activation=None)

    # Compute predictions.
    if mode == estimator.ModeKeys.PREDICT:
        predictions = {"predictions_{}".format(l): phi[:, i]
                       for i, l in enumerate(metadata.targets.labels)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Use a loss for training
    ll_f = tf.compat.v1.distributions.Normal(loc=phi, scale=1.0)
    loss = -1 * tf.reduce_mean(input_tensor=ll_f.log_prob(Y))
    tf.compat.v1.summary.scalar("loss", loss)

    # Compute evaluation metrics.
    mse = tf.compat.v1.metrics.mean_squared_error(labels=Y, predictions=phi)
    metrics = {"mse": mse}

    if mode == estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)

    # For training, use Adam to learn
    assert mode == estimator.ModeKeys.TRAIN
    optimizer = tf.compat.v1.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
