"""Generic classification config file."""
import aboleth as ab
import tensorflow as tf

from landshark.model import patch_categories, patch_slices

ab.set_hyperseed(666)
embed_dim = 3

def model(features, labels, mode, params):

    metadata = params["metadata"]
    N = metadata.features.N
    n_samples = 3

    kernel = ab.RBF(10.0, learn_lenscale=True)
    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.RandomFourier(n_features=32, kernel=kernel) >>
        ab.DenseVariational(output_dim=1, full=True, prior_std=1.0,
                            learn_prior=True)
    )

    phi, kl = net(X=features["ord"])
    std = ab.pos_variable(10.0, name="noise")
    ll_f = tf.distributions.Normal(loc=phi, scale=std)
    predict_mean = ab.sample_mean(phi)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': predict_mean,
            'sample_1': phi[0],
            'sample_2': phi[1],
            'sample_3': phi[2]
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ll = ll_f.log_prob(labels)
    loss = ab.elbo(ll, kl, N)
    tf.summary.scalar('loss', loss)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=labels,
                                        predictions=predict_mean,
                                        name='mse_op')
    metrics = {'mse': mse}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

