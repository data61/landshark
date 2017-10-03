import logging
from itertools import product, islice, tee

# import tensorflow as tf
import numpy as np
import landshark as ls
#  TODO do this in the __init__
import landshark.image
import landshark.patch
import tables
import aboleth as ab
from typing import Tuple, Iterator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# from tensorflow.contrib.data import Dataset

# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

halfwidth = 1
patchwidth = (2 * halfwidth) + 1
target_label = 'Na_ppm_i_1'
# target_label = 'Na_ppm_imp'


# Aboleth settings
l_samples = 5
p_samples = 10
rseed = 666
frac_test = 0.1
n_epochs = 2000
bsize = 50
# config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?
variance = 10.0

batchsize = 10000

fake_width = 400
fake_height = 400


def sk_validate(Y: np.ndarray, Xiter: Iterator[Tuple[np.ndarray, np.ndarray]])\
        -> RandomForestRegressor:

    Xord, Xcat = zip(*Xiter)
    Xo = np.ma.concatenate(Xord, axis=0).reshape((len(Y), -1))
    N, D = Xo.shape

    # Split the training and testing data
    X_tr, X_ts, Y_tr, Y_ts, M_tr, M_ts = train_test_split(
        Xo.data.astype(np.float32),
        Y.astype(np.float32),
        Xo.mask,
        test_size=frac_test,
        random_state=rseed
        )
    # X_tr, X_ts, Y_tr, Y_ts = train_test_split(
    #     Xo.data.astype(np.float32),
    #     Y.astype(np.float32),
    #     test_size=frac_test,
    #     random_state=rseed
    #     )
    # N_tr, D = X_tr.shape

    # Means should be zero, so this is a mean impute
    X_tr[M_tr] = 0.
    X_ts[M_ts] = 0.

    rf = RandomForestRegressor(n_estimators=10)
    #rf = LinearRegression()
    rf.fit(X_tr, Y_tr)
    Ey = rf.predict(X_ts)
    r2 = r2_score(Y_ts.flatten(), Ey.flatten())
    print("Random Forest r2: {}".format(r2))

    return rf


def ab_validate(Y: np.ndarray, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray) \
        -> None:
    """Train and validate an aboleth model."""
    N, D = Xo.shape

    l = tf.Variable(1. * np.ones((D, 1), dtype=np.float32))
    kernel = ab.RBF(lenscale=ab.pos(l))
    layers = ab.LearnedNormalImpute(
        ab.InputLayer('Xo', n_samples=l_samples),
        ab.InputLayer('Xo_m')
        ) >> \
        ab.RandomFourier(n_features=100, kernel=kernel) >> \
        ab.DenseVariational(output_dim=1, full=True)

    # Split the training and testing data
    X_tr, X_ts, Y_tr, Y_ts, M_tr, M_ts = train_test_split(
        Xo.data.astype(np.float32),
        Y.astype(np.float32)[:, np.newaxis],
        Xo.mask,
        test_size=frac_test,
        random_state=rseed
        )
    N_tr, D = X_tr.shape

    # Data
    with tf.name_scope("Input"):
        Xb, Yb, Mb = batch_training(X_tr, Y_tr, M_tr, n_epochs=n_epochs,
                                    batch_size=bsize)
        X_ = tf.placeholder_with_default(Xb, shape=(None, D))
        Y_ = tf.placeholder_with_default(Yb, shape=(None, 1))
        M_ = tf.placeholder_with_default(Mb, shape=(None, D))

    with tf.name_scope("Deepnet"):
        lkhood = ab.likelihoods.Normal(variance=ab.pos(tf.Variable(variance)))
        nn, kl = layers(Xo=X_, Xo_m=M_)
        loss = ab.elbo(nn, Y_, N_tr, kl, lkhood)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Logging learning progress
    log = tf.train.LoggingTensorHook(
        {'step': global_step, 'loss': loss},
        every_n_iter=1000
        )

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=config,
            save_summaries_steps=None,
            save_checkpoint_secs=None,
            hooks=[log]) as sess:
        try:
            while not sess.should_stop():
                sess.run(train)
        except tf.errors.OutOfRangeError:
            print('Input queues have been exhausted!')
            pass

        # Prediction
        feed_dict = {X_: X_ts, Y_: [[None]], M_: M_ts}
        # Prediction, the [[None]] is to stop the default placeholder queue
        # TODO: call n_groups something like sample_multiplier
        Eys = ab.predict_samples(nn, feed_dict=feed_dict, n_groups=p_samples,
                                 session=sess)

    Ey = Eys.mean(axis=0)
    r2 = r2_score(Y_ts.flatten(), Ey.flatten())
    print("Aboleth r2: {}".format(r2))
    # import IPython; IPython.embed()


def predict(model, X_it):
    for x in X_it:
        Xs = x[0].data
        Xs[x[0].mask] = 0.  # impute
        Xs = Xs.reshape((len(Xs), -1))
        ys = model.predict(Xs)
        # print(np.max(ys), np.min(ys), np.any(np.isinf(ys)),
        #       np.any(np.isnan(ys)))
        yield ys


def show(Y_it, xfile):
    image_height = xfile.root._v_attrs.height
    image_width = xfile.root._v_attrs.width
    # image_width, image_height = fake_width, fake_height
    Y = np.concatenate(list(Y_it))
    im = Y.reshape((image_height, image_width))
    # im = Y.reshape((100, 100))
    import matplotlib.pyplot as pl
    from matplotlib import cm
    pl.imshow(im, cmap=cm.inferno)
    pl.show()


def batch_training(X, Y, M, batch_size, n_epochs):
    """Batch training queue convenience function."""
    data_tr = Dataset.from_tensor_slices({'X': X, 'Y': Y, 'M': M}) \
        .shuffle(buffer_size=1000, seed=rseed) \
        .repeat(n_epochs) \
        .batch(batch_size)
    data = data_tr.make_one_shot_iterator().get_next()
    return data['X'], data['Y'], data['M']


if __name__ == "__main__":
    xfile = tables.open_file("lbalpha.hdf5")
    yfile = tables.open_file("geochem_sites.hdf5")
    ord_cache = RowCache(xfile.root.ordinal_data, 20, 20)
    cat_cache = RowCache(xfile.root.categorical_data, 20, 20)

    train_coords_it, Y = read_targets(xfile, yfile)
    train_X_it = read_features(xfile, ord_cache, cat_cache, train_coords_it)
    predict_X_it = read_features(xfile, ord_cache, cat_cache)

    # ab_validate(Y, ord_data, cat_data)
    model = sk_validate(Y, train_X_it)
    predict_Y_it = predict(model, predict_X_it)
    # for i, y in enumerate(predict_Y_it):
    #     print(i)
    #     if i > 5:
    #         break
    show(predict_Y_it, xfile)
