"""Train/test with tfrecords."""
import signal
import logging
import os.path
import pickle
from itertools import count

import numpy as np
import tensorflow as tf
import aboleth as ab

from landshark import config as cf

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


fdict = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }


def dataset(records, batch_size: int, testing=False):
    """Train and test."""
    if not testing:
        dataset = tf.data.TFRecordDataset(records).repeat(count=cf.epochs) \
            .shuffle(buffer_size=1000).batch(batch_size)
    else:
        dataset = tf.data.TFRecordDataset(records).batch(batch_size) \
            .interleave(
                lambda x: tf.data.Dataset.from_tensors(x).repeat(cf.psamps),
                cycle_length=1,
                block_length=cf.psamps
                )
    return dataset


def decode(iterator, record_shape):
    str_features = iterator.get_next()
    raw_features = tf.parse_example(str_features, features=fdict)
    npatch = (2 * record_shape.halfwidth + 1) ** 2
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

        # Placeholders for prediction
        xo_ = tf.placeholder_with_default(x_ord, x_ord.shape, name="Xo")
        xom_ = tf.placeholder_with_default(x_ord_mask, x_ord.shape, name="Xom")
        xc_ = tf.placeholder_with_default(x_cat, x_cat.shape, name="Xc")
        xcm_ = tf.placeholder_with_default(x_cat_mask, x_cat.shape, name="Xcm")
        y_ = tf.placeholder_with_default(y, y.shape, name="Y")

    return xo_, xom_, xc_, xcm_, y_


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def predict_dict(data, Xo, Xom, Xc, Xcm):
    for d in data:
        N = len(d.x_ord)
        xord = np.ma.reshape(d.x_ord, [N, -1])
        xcat = np.ma.reshape(d.x_cat, [N, -1])
        fdict = {Xo: xord.data, Xom: xord.mask, Xc: xcat.data, Xcm: xcat.mask}
        yield fdict


def predict(model, data):

    model_file = tf.train.latest_checkpoint(model)
    print("Loading model: {}".format(model_file))

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=cf.predict_config)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(model_file))
            saver.restore(sess, model_file)

            # Restore place holders and prediction network
            Xo = graph.get_operation_by_name("Inputs/Xo").outputs[0]
            Xom = graph.get_operation_by_name("Inputs/Xom").outputs[0]
            Xc = graph.get_operation_by_name("Inputs/Xc").outputs[0]
            Xcm = graph.get_operation_by_name("Inputs/Xcm").outputs[0]
            phi = graph.get_operation_by_name("Deepnet/nnet").outputs[0]
            # TODO plus noise

            for i, d in enumerate(predict_dict(data, Xo, Xom, Xc, Xcm)):
                log.info("predicting batch {}".format(i))
                y_samples = ab.predict_samples(phi, d, cf.psamps, sess)
                Ey = y_samples.mean(axis=0)
                Sf = y_samples.std(axis=0)
                yield Ey, Sf


def train_test(records_train, records_test, train_metadata, test_metadata,
               name):

    train_dataset = dataset(records_train, cf.batch_size, testing=False)
    test_dataset = dataset(records_test, cf.batch_size, testing=True)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xo, Xom, Xc, Xcm, Y = decode(iterator, train_metadata)
    N = train_metadata.N

    with tf.name_scope("Deepnet"):
        phi, lkhood, loss = cf.model(Xo, Xom, Xc, Xcm, Y, N)
        phi = tf.identity(phi, name="nnet")
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
    r2_score = None

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=cf.train_config,
            checkpoint_dir=checkpoint_dir,
            scaffold=tf.train.Scaffold(local_init_op=train_init_op),
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            hooks=[logger]
            ) as sess:

        for i in count():
            log.info("Training round {} with {} epochs.".format(i, cf.epochs))
            try:

                # Train loop
                try:
                    while not sess.should_stop():
                        _, g = sess.run([train, global_step])
                except tf.errors.OutOfRangeError:
                    log.info("Training epoch complete.")

                # Test loop
                sess.run(test_init_op)
                Ys, EYs, = [], []
                try:
                    while not sess.should_stop():
                        samples = []
                        for j in range(cf.psamps):
                            y, ey = sess.run([Y, phi])
                            samples.append(ey)

                        Ys.append(y)
                        cat_samples = np.concatenate(samples, axis=0)
                        EYs.append(cat_samples.mean(axis=0))
                except tf.errors.OutOfRangeError:
                    pass

                # Scores
                Ys = np.concatenate(Ys)
                EYs = np.concatenate(EYs)
                r2_score = rsquare(Ys, EYs)
                rsquare_summary(r2_score, sess, g)
                log.info("Aboleth r2: {:.4f}".format(r2_score))

            except KeyboardInterrupt:
                log.info("Training ended, final R-square = {:.4f}."
                         .format(r2_score))
                break

            sess.run(train_init_op)

    return checkpoint_dir


def rsquare(Y: np.ndarray, EY: np.ndarray) -> float:
    # assert len(Y) == len(stats)
    # Y = np.concatenate([Y[j] for j in range(len(Y))])
    # EY = np.concatenate([stats[j][0] for j in range(len(stats))])

    SS_ref = np.sum((Y - EY)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = float(1 - SS_ref / SS_tot)
    return R2


def rsquare_summary(r2_score, session, step=None):
    # Get a summary writer for R-square
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag='r-square', simple_value=r2_score)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def show(pred, image_spec):
    image_height = image_spec.height
    image_width = image_spec.width
    Y, _ = zip(*pred)
    Y = np.concatenate(Y).squeeze()
    im = Y.reshape((image_height, image_width))
    import matplotlib.pyplot as pl
    from matplotlib import cm
    pl.imshow(im, cmap=cm.inferno)
    pl.show()
