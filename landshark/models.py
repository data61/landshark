"""Models."""
import os
import logging
import pickle
from collections import namedtuple

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger(__name__)

frac_test = 0.1
rseed = 666
n_epochs = 2
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU? 0 is no


def train_tf(data):

    X_ord, X_cat, Y = batch_training(data, batch_size, n_epochs)
    cat = tf.concat((X_ord, tf.to_float(X_cat)), axis=1)

    # Logging learning progress
    # logger = tf.train.LoggingTensorHook(
    #     {'step': global_step, 'loss': loss},
    #     every_n_iter=1000
    # )

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=config,
            save_summaries_steps=None,
            save_checkpoint_secs=None,
            hooks=[]
            ) as sess:
        try:
            i = 0
            while not sess.should_stop():
                res = sess.run(cat)
                print(i, res.shape)
                i += 1
        except tf.errors.OutOfRangeError:
            print('Input queues have been exhausted!')
            pass

    return None


def batch_training(data, batch_size, n_epochs):
    """Batch training queue convenience function."""
    data_tr = tf.data.Dataset.from_generator(data, data.types, data.shapes) \
        .shuffle(buffer_size=100, seed=rseed) \
        .batch(batch_size)
    batches = data_tr.make_one_shot_iterator().get_next()
    return batches


def train(data):
    X_ord_list = []
    X_cat_list = []
    Y_list = []
    for d in data:
        X_ord_list.append(d.x_ord)
        X_cat_list.append(d.x_cat)
        Y_list.append(d.y)
    X_ord = np.ma.concatenate(X_ord_list, axis=0)
    X_cat = np.ma.concatenate(X_cat_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    Xo = np.ma.concatenate(X_ord, axis=0).reshape((len(Y), -1))
    N, D = Xo.shape

    # Split the training and testing data
    X_tr, X_ts, Y_tr, Y_ts, M_tr, M_ts = train_test_split(
        Xo.data, Y, Xo.mask, test_size=frac_test, random_state=rseed)

    # Means should be zero, so this is a mean impute
    X_tr[M_tr] = 0.
    X_ts[M_ts] = 0.

    rf = RandomForestRegressor(n_estimators=10)
    #rf = LinearRegression()
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

