import logging

import tensorflow as tf
import numpy as np
import landshark as ls
#  TODO do this in the __init__
import landshark.image
import landshark.patch
import tables
import aboleth as ab
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.contrib.data import Dataset

# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

halfwidth = 2
patch_pixels = ((2 * halfwidth) + 1) ** 2
target_label = 'Na_ppm_i_1'
# target_label = 'Na_ppm_imp'

xfile = tables.open_file("lbalpha.hdf5")
yfile = tables.open_file("geochem_sites.hdf5")

# Aboleth settings
l_samples = 5
p_samples = 10
rseed = 666
frac_test = 0.1
n_epochs = 300
bsize = 50
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?
variance = 10.0


def read_data() -> Tuple[np.ndarray, np.ma.MaskedArray, np.ma.MaskedArray]:
    #  TODO iterate these properly in batches
    #  TODO save these in the opposite order so dont transpose
    #  TODO sort the coordinates in the Y
    #  TODO find n_categories at image read time
    coords_x, coords_y = yfile.root.coordinates.read().transpose()
    targets = yfile.root.targets.read()
    labels = yfile.root.targets.attrs.labels
    Y = targets[:, labels.index(target_label)]
    n = targets.shape[0]

    x_pixel_array = xfile.root.x_coordinates.read()
    y_pixel_array = xfile.root.y_coordinates.read()
    bounds = ls.image.bounds(x_pixel_array, y_pixel_array)
    image_height = xfile.root._v_attrs.height
    image_width = xfile.root._v_attrs.width

    data_in_bounds = ls.image.in_bounds(coords_x, coords_y, bounds)
    assert np.all(data_in_bounds)

    coords_x_image = ls.image.world_to_image(coords_x, x_pixel_array)
    coords_y_image = ls.image.world_to_image(coords_y, y_pixel_array)

    ord_data = xfile.root.ordinal_data
    cat_data = xfile.root.categorical_data

    # TODO could make these at batch time if there are billions of points
    patches = [ls.patch.Patch(x, y, halfwidth, image_width, image_height)
               for x, y in zip(coords_x_image, coords_y_image)]

    ord_patch_data = np.empty((n, patch_pixels, ord_data.shape[1]),
                              dtype=np.float32)
    cat_patch_data = np.empty((n, patch_pixels, cat_data.shape[1]),
                              dtype=np.int32)

    for i, p in enumerate(patches):
        #  iterating over contiguous reads for a patch
        for rp, r in zip(p.patch_flat, p.flat):
            ord_patch_data[i, rp] = ord_data[r]
            cat_patch_data[i, rp] = cat_data[r]

    # TODO missing data harder if everything flat
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

    ord_flat = ord_marray.reshape((ord_marray.shape[0], -1))
    cat_flat = cat_marray.reshape((cat_marray.shape[0], -1))

    ord_flat -= ord_flat.mean(axis=0)
    ord_flat /= ord_flat.std(axis=0)

    Y -= Y.mean()
    Y /= Y.std()

    return Y, ord_flat, cat_flat


def validate(Y: np.ndarray, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray) \
        -> None:
    """Train and validate an aboleth model."""
    N, D = Xo.shape

    # import IPython; IPython.embed()

    layers = ab.LearnedNormalImpute(
        ab.InputLayer('Xo', n_samples=l_samples),
        ab.InputLayer('Xo_m')
        ) >> \
        ab.DenseVariational(output_dim=1)

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
        # import IPython; IPython.embed(); exit()
        loss = ab.elbo(nn, Y_, N_tr, kl, lkhood)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(0.001)
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
    print("r2: {}".format(r2))
    import IPython; IPython.embed()


def batch_training(X, Y, M, batch_size, n_epochs):
    """Batch training queue convenience function."""
    data_tr = Dataset.from_tensor_slices({'X': X, 'Y': Y, 'M': M}) \
        .shuffle(buffer_size=1000, seed=rseed) \
        .repeat(n_epochs) \
        .batch(batch_size)
    data = data_tr.make_one_shot_iterator().get_next()
    return data['X'], data['Y'], data['M']


if __name__ == "__main__":
    Y, ord_data, cat_data = read_data()
    validate(Y, ord_data, cat_data)
