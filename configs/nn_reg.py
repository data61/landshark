"""Generic classification config file."""
import tensorflow as tf

from landshark.config import flatten_patch

EMBED_DIM = 3

def model(features, labels, mode, params):
    """
    Describe the specification of a Tensorflow custom estimator model.

    This function must be implemented in all configurations. It is exactly
    the model function passed to a custom Tensorflow estimator.
    See https://www.tensorflow.org/guide/custom_estimators

    Parameters
    ----------
    features : dict
        Features is a recursive dictionary of tensors, proving the X inputs
        for the model (from the images). The dictonary has the following
        entries:
            indices -- (?, 2) the image coordinates of features
            coords -- (?, 2) the world coordinates (x, y) of features
            con -- dict of continuous feature columns
            cat -- dict of categorical feature columns
        Each con and cat feature is itself a dict, with two items:
            data -- the column data tensor
            mask -- the mask tensor
        The data and mask tensors are always of shape (?, p, p, 1)
        where p is the patch side length.
    labels : tf.Tensor
        A (?, k) tensor giving the k targets for the prediction.
    mode : tf.estimator.ModeSpec
        One of TRAIN, TEST or EVAL, describing in which context this code
        is being run by the estimator.
    params : dict
        Extra params given by the estimator. The critical one for configs
        is "metadata" that has comprehensive information about the features
        and targets useful for model building (for example, the number of
        possible values for each categorical column). For more details
        check the Landshark documentation.

    Returns
    -------
    tf.EstimatorSpec
        An EstimatorSpec object describing the model. For details check
        the Tensorflow custom estimator howto.

    """

    metadata = params["metadata"]
    N = metadata.features.N

    # let's 0-impute continuous columns
    def zero_impute(x, m):
        r = x * tf.cast(tf.logical_not(m), x.dtype)
        return r
    con_cols = {k: zero_impute(v["data"], v["mask"])
                for k, v in features["con"].items()}
    # zero is the missing value so we can use it as extra category
    cat_cols = {k: v["data"] for k, v in features["cat"].items()}

    # just concatenate the patch pixels as more features
    con_cols = {k: flatten_patch(v) for k, v in con_cols.items()}
    cat_cols = {k: flatten_patch(v) for k, v in cat_cols.items()}

    # For simplicity, use the tensorflow feature columns.
    columns_con = [tf.feature_column.numeric_column(k)
                   for k in con_cols.keys()]
    columns_cat = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
        key=k, num_buckets=(v + 1)), EMBED_DIM)
        for k, v in zip(metadata.features.categorical.labels,
        metadata.features.categorical.ncategories)]

    inputs_con = tf.feature_column.input_layer(con_cols, columns_con)
    inputs_cat = tf.feature_column.input_layer(cat_cols, columns_cat)
    inputs = tf.concat([inputs_con, inputs_cat], axis=1)

    # Build a simple 2-layer network
    l1 = tf.layers.dense(inputs, units=64, activation=tf.nn.relu)
    l2 = tf.layers.dense(l1, units=32, activation=tf.nn.relu)

    # Get some predictions for the labels
    phi = tf.layers.dense(l2, units=labels.shape[1], activation=tf.nn.relu)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions_{}'.format(l): phi[:, i]
                       for i, l in enumerate(metadata.targets.labels)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Use a loss for training
    ll_f = tf.distributions.Normal(loc=phi, scale=1.0)
    loss = -1 * tf.reduce_mean(ll_f.log_prob(labels))
    tf.summary.scalar('loss', loss)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=labels,
                                        predictions=phi)
    metrics = {'mse': mse}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # For training, use Adam to learn
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
