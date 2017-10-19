"""Train/test with tfrecords."""

import logging
import os.path
import numpy as np
from typing import List
import tensorflow as tf
import aboleth as ab
import pickle

rseed = 666
batch_size = 10
psamps = 10
nsamps = 5
train_config = tf.ConfigProto(device_count={"GPU": 0})
predict_config = tf.ConfigProto(device_count={"GPU": 0})
epochs = 20

log = logging.getLogger(__name__)

fdict = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }


def batch(records: List[str], record_shape, batch_size: int, epochs: int):
    """Train and test."""
    npatch = (2 * record_shape.halfwidth + 1) ** 2
    dataset = tf.data.TFRecordDataset(records).repeat(count=epochs) \
        .shuffle(buffer_size=100).batch(batch_size)
    iterator = dataset.make_one_shot_iterator().get_next()
    raw_features = tf.parse_example(iterator, features=fdict)

    with tf.name_scope("Inputs"):
        x_ord = tf.decode_raw(raw_features["x_ord"], tf.float32)
        x_cat = tf.decode_raw(raw_features["x_cat"], tf.int32)
        x_ord_mask = tf.decode_raw(raw_features["x_ord_mask"], tf.uint8)
        x_cat_mask = tf.decode_raw(raw_features["x_cat_mask"], tf.uint8)
        x_ord_mask = tf.cast(x_ord_mask, tf.bool)
        x_cat_mask = tf.cast(x_cat_mask, tf.bool)
        y = tf.decode_raw(raw_features["y"], tf.float32)

        x_ord.set_shape((None, npatch * record_shape.x_ord))
        x_ord_mask.set_shape((None, npatch * record_shape.x_ord))
        x_cat.set_shape((None, npatch * record_shape.x_cat))
        x_cat_mask.set_shape((None, npatch * record_shape.x_cat))
        y.set_shape((None, 1))

    return x_ord, x_ord_mask, x_cat, x_cat_mask, y


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def train_test(records_train, records_test,
               train_metadata, test_metadata, name):

    Xof, Xomf, Xcf, Xcmf, Y = batch(records_train, train_metadata,
                                    batch_size, epochs)

    lenscale = np.ones((Xof.shape[1], 1), dtype=np.float32) * 10.

    data_input = ab.InputLayer(name="X", n_samples=nsamps)  # Data input
    mask_input = ab.MaskInputLayer(name="M")  # Missing data mask input
    kern = ab.RBF(lenscale=ab.pos(tf.Variable(lenscale)))
    net = (
        ab.LearnedScalarImpute(data_input, mask_input) >>
        ab.RandomFourier(n_features=50, kernel=kern) >>
        ab.DenseVariational(output_dim=1, std=1., full=True)
        )

    # This is where we build the actual GP model
    with tf.name_scope("Deepnet"):
        N = train_metadata.N
        phi, kl = net(X=Xof, M=Xomf)
        phi = tf.identity(phi, name="nnet")
        noise = tf.Variable(1.0)
        lkhood = tf.distributions.StudentT(df=5., loc=phi, scale=ab.pos(noise))
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
            config=train_config,
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

    # Ey, Sf, Y_s = zip(*predict_tf(checkpoint_dir, data_test))
    # Ey = np.vstack(Ey).squeeze()
    # Y_s = np.vstack(Y_s).squeeze()
    # r2 = r2_score(Y_s, Ey)
    # log.info("Aboleth r2: {}".format(r2))

    return checkpoint_dir
