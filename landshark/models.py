"""Models."""
import os
import logging
import pickle
from itertools import chain
from collections import namedtuple

from typing import Iterator, Tuple
import tensorflow as tf
import numpy as np
import aboleth as ab
from sklearn.metrics import r2_score
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from landshark.feed import TrainingBatch

log = logging.getLogger(__name__)

rseed = 666
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 1})  # Use GPU? 0 is no


class SliceTrainingData:

    def __init__(self, data: Iterator[TrainingBatch]) -> None:
        peek_d = next(data)

        self.types = (
            tf.as_dtype(peek_d.x_ord.dtype),  # ord data
            tf.bool,  # ord mask
            tf.as_dtype(peek_d.x_cat.dtype),  # cat data
            tf.bool,  # cat mask
            tf.as_dtype(peek_d.y.dtype)  # target
            )

        self.shapes = (
            peek_d.x_ord.shape[1:],
            peek_d.x_ord.shape[1:],
            peek_d.x_cat.shape[1:],
            peek_d.x_cat.shape[1:],
            peek_d.y.shape[1:]
            )

        self.data = chain([peek_d], data)

    def __call__(self) -> Iterator[Tuple[np.array, np.array, np.array,
                                         np.array, np.array]]:
        for d in self.data:
            for xo, xc, y in zip(d.x_ord, d.x_cat, d.y):
                tslice = (xo.data, xo.mask, xc.data, xc.mask, y)
                yield tslice


def extract_masks_test(data):
    gen = (
        (d.x_ord.data,
         d.x_ord.mask,
         d.x_cat.data,
         d.x_cat.mask,
         d.y) for d in data)
    return gen


def extract_masks_query(data):
    gen = (
        (d.x_ord.data,
         d.x_ord.mask,
         d.x_cat.data,
         d.x_cat.mask) for d in data)
    return gen


def batch_training(data, batch_size):
    """Batch training queue convenience function."""

    # Make the training data iterator
    data_tr = tf.data.Dataset.from_generator(data, data.types, data.shapes) \
        .shuffle(buffer_size=100, seed=rseed) \
        .batch(batch_size)
    batches = data_tr.make_one_shot_iterator().get_next()

    # Make placeholders for prediction
    with tf.name_scope("Inputs"):
        Xo = tf.placeholder_with_default(batches[0], (None,) + data.shapes[0],
                                         name="Xo")
        Xom = tf.placeholder_with_default(batches[1], (None,) + data.shapes[1],
                                          name="Xom")
        Xc = tf.placeholder_with_default(batches[2], (None,) + data.shapes[2],
                                         name="Xc")
        Xcm = tf.placeholder_with_default(batches[3], (None,) + data.shapes[3],
                                          name="Xcm")
        Y = tf.placeholder_with_default(batches[4], (None,) + data.shapes[4],
                                        name="Y")

    return Xo, Xom, Xc, Xcm, Y


def train_tf(data_train, data_test, name):

    datgen = SliceTrainingData(data_train)
    Xo, Xom, Xc, Xcm, Y = batch_training(datgen, batch_size)

    # Get training data
    # TODO, stream this
    Xo_s = []
    Xom_s = []
    Y_s = []
    for xo, xom, _, _, y in extract_masks_test(data_test):
        Xo_s.append(xo)
        Xom_s.append(xom)
        Y_s.append(y)
    Xo_s = np.vstack(Xo_s)
    Xom_s = np.vstack(Xom_s)
    Y_s = np.vstack(Y_s)

    data_input = ab.InputLayer(name="X", n_samples=1)  # Data input
    mask_input = ab.MaskInputLayer(name="M")  # Missing data mask input
    kern = ab.RBF(lenscale=ab.pos(tf.Variable(10.)))
    net = (
        ab.LearnedScalarImpute(data_input, mask_input) >>
        ab.RandomFourier(n_features=50, kernel=kern) >>
        ab.DenseVariational(output_dim=1, std=1., full=True)
        )

    # This is where we build the actual GP model
    with tf.name_scope("Deepnet"):
        N = round(1026 * 0.9)
        phi, kl = net(X=Xo, M=Xom)
        phi = tf.identity(phi, name="nnet")
        noise = tf.Variable(1.)
        lkhood = tf.distributions.Normal(loc=phi, scale=ab.pos(noise))
        lkhood = tf.identity(lkhood, name="lkhood")
        loss = ab.elbo(lkhood, Y, N, kl)
        tf.summary.scalar("loss", loss)

    # Set up the trainig graph
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Logging learning progress
    logger = tf.train.LoggingTensorHook(
        {"step": global_step, "loss": loss},
        every_n_iter=100
        )

    checkpoint_dir = os.path.join(os.getcwd(), name)

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=config,
            checkpoint_dir=checkpoint_dir,
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            hooks=[logger]
            ) as sess:
        try:
            while not sess.should_stop():
                sess.run(train)
        except tf.errors.OutOfRangeError:
            log.info("Input queues have been exhausted!")
            pass

        feed_dict = {Xo: Xo_s, Xom: Xom_s, Y: [[None]]}
        Ey = ab.predict_expected(phi, feed_dict=feed_dict, n_groups=10,
                                 session=sess)

    r2 = r2_score(Y_s.flatten(), Ey.flatten())
    log.info("Aboleth r2: {}".format(r2))

    return checkpoint_dir


def train(data_train, data_test):

    def cat_data(data):
        X_ord_list = []
        X_cat_list = []
        Y_list = []
        for d in data:
            X_ord_list.append(d.x_ord)
            X_cat_list.append(d.x_cat)
            Y_list.append(d.y)
        Y = np.concatenate(Y_list, axis=0)
        X_ord = np.ma.concatenate(X_ord_list, axis=0).reshape((len(Y), -1))
        X_cat = np.ma.concatenate(X_cat_list, axis=0).reshape((len(Y), -1))
        X_ord.data[X_ord.mask] = 0  # 0 mean impute
        X_cat.data[X_cat.mask] = 0
        X = np.concatenate((X_ord.data, X_cat.data), axis=1)
        return Y, X

    Y_tr, X_tr = cat_data(data_train)
    Y_ts, X_ts = cat_data(data_test)

    # rf = RandomForestRegressor(n_estimators=10)
    rf = LinearRegression()
    log.info("Training random forest")
    rf.fit(X_tr, Y_tr)
    Ey = rf.predict(X_ts)
    r2 = r2_score(Y_ts.flatten(), Ey.flatten())
    log.info("Random Forest r2: {}".format(r2))
    return rf


Model = namedtuple("Model", ['skmodel', 'halfwidth', 'y_label'])


def write(model, halfwidth, y_label, name):
    path = os.path.join(os.getcwd(), name + ".lsmodel")
    m = Model(skmodel=model, halfwidth=halfwidth, y_label=y_label)
    with open(path, 'wb') as f:
        log.info("Writing model to disk")
        pickle.dump(m, f)


def load(fname):
    with open(fname, 'rb') as f:
        log.info("Loading model from disk")
        m = pickle.load(f)
    return m


def predict(model, X_it):
    for x in X_it:
        Xs = x[0].data
        Xs[x[0].mask] = 0.  # impute
        Xs = Xs.reshape((len(Xs), -1))
        ys = model.predict(Xs)
        yield ys


# def predict_tf(model, data_test):


def show(Y_it, image_spec):
    image_height = image_spec.height
    image_width = image_spec.width
    Y = np.concatenate(list(Y_it))
    im = Y.reshape((image_height, image_width))
    # im = Y.reshape((100, 100))
    import matplotlib.pyplot as pl
    from matplotlib import cm
    pl.imshow(im, cmap=cm.inferno)
    pl.show()
