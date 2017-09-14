import logging
from itertools import product, islice

import tensorflow as tf
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
from tensorflow.contrib.data import Dataset

# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

halfwidth = 1
patchwidth = (2 * halfwidth) + 1
patch_pixels = patchwidth ** 2
target_label = 'Na_ppm_i_1'
# target_label = 'Na_ppm_imp'


# Aboleth settings
l_samples = 5
p_samples = 10
rseed = 666
frac_test = 0.1
n_epochs = 2000
bsize = 50
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?
variance = 10.0

batchsize = 10000


def get_coords_training(coords, x_pixel_array, y_pixel_array):
    n = coords.shape[0]
    c = 0
    while c < n:
        start = c
        stop = min(c + batchsize, n)
        out = coords[start:stop].transpose()
        c += batchsize
        coords_x, coords_y = out
        im_coords_x = ls.image.world_to_image(coords_x, x_pixel_array)
        im_coords_y = ls.image.world_to_image(coords_y, y_pixel_array)
        yield im_coords_x, im_coords_y


def get_coords_query(image_width, image_height):
    coords_it = product(range(image_width), range(image_height))
    while True:
        out = list(islice(coords_it, batchsize))
        if len(out) == 0:
            return
        else:
            coords_x, coords_y = zip(*out)
            cx = np.array(coords_x)
            cy = np.array(coords_y)
            yield cx, cy


def read_batch(coords_x, coords_y, hfile):
    image_height = hfile.root._v_attrs.height
    image_width = hfile.root._v_attrs.width
    ord_data = hfile.root.ordinal_data
    cat_data = hfile.root.categorical_data
    n = coords_x.shape[0]
    n_feats_ord = ord_data.shape[1]
    n_feats_cat = cat_data.shape[1]
    patches = [ls.patch.Patch(x, y, halfwidth, image_width, image_height)
               for x, y in zip(coords_x, coords_y)]
    ord_patch_data = np.empty((n, patch_pixels, n_feats_ord),
                              dtype=np.float32)
    cat_patch_data = np.empty((n, patch_pixels, n_feats_cat),
                              dtype=np.int32)

    for i, p in enumerate(patches):
        #  iterating over contiguous reads for a patch
        for rp, r in zip(p.patch_flat, p.flat):
            ord_patch_data[i, rp] = ord_data[r]
            cat_patch_data[i, rp] = cat_data[r]

    cat_missing = cat_data.attrs.missing_values
    ord_missing = ord_data.attrs.missing_values

    ord_mask = np.zeros_like(ord_patch_data, dtype=bool)
    cat_mask = np.zeros_like(cat_patch_data, dtype=bool)

    for i, v in enumerate(cat_missing):
        if v is not None:
            cat_mask[:, :, i] = cat_patch_data[:, :, i] == v

    for i, v in enumerate(ord_missing):
        if v is not None:
            ord_mask[:, :, i] = ord_patch_data[:, :, i] == v

    ord_marray = np.ma.MaskedArray(data=ord_patch_data, mask=ord_mask)
    cat_marray = np.ma.MaskedArray(data=cat_patch_data, mask=cat_mask)

    return ord_marray, cat_marray


def read_targets(xfile, yfile):
    x_pixel_array = xfile.root.x_coordinates.read()
    y_pixel_array = xfile.root.y_coordinates.read()
    coords_it = get_coords_training(yfile.root.coordinates, x_pixel_array,
                                    y_pixel_array)
    labels = yfile.root.targets.attrs.labels
    targets = yfile.root.targets.read()
    Y = targets[:, labels.index(target_label)]
    return coords_it, Y


def read_features(xfile, coords_it=None):
    image_height = xfile.root._v_attrs.height
    image_width = xfile.root._v_attrs.width
    if coords_it is None:
        coords_it = get_coords_query(image_width, image_height)
        # coords_it = get_coords_query(100, 100)
    data_batches = (read_batch(cx, cy, xfile) for cx, cy in coords_it)
    return data_batches


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
    N_tr, D = X_tr.shape

    # Means should be zero, so this is a mean impute
    X_tr[M_tr] = 0.
    X_ts[M_ts] = 0.

    rf = RandomForestRegressor(n_estimators=10)
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
        print(np.max(ys), np.min(ys), np.any(np.isinf(ys)),
              np.any(np.isnan(ys)))
        yield ys


def show(Y_it, xfile):
    image_height = xfile.root._v_attrs.height
    image_width = xfile.root._v_attrs.width

    Y = np.concatenate(list(Y_it))
    im = Y.reshape((image_height, image_width))
    # im = Y.reshape((100, 100))
    import matplotlib.pyplot as pl
    pl.imshow(im)
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
    train_coords_it, Y = read_targets(xfile, yfile)
    train_X_it = read_features(xfile, train_coords_it)
    predict_X_it = read_features(xfile)
    # import IPython; IPython.embed()

    # ab_validate(Y, ord_data, cat_data)
    model = sk_validate(Y, train_X_it)
    predict_Y_it = predict(model, predict_X_it)
    show(predict_Y_it, xfile)
