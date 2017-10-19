"""Train/test with tfrecords."""
import signal
import logging
import os.path
import pickle
from itertools import count

import numpy as np
import tensorflow as tf
import aboleth as ab

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)

# FIXME Setting, should go in a config
rseed = 666
batch_size = 10
psamps = 10
nsamps = 5
train_config = tf.ConfigProto(device_count={"GPU": 0})
predict_config = tf.ConfigProto(device_count={"GPU": 0})

epochs = 10

fdict = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }


def dataset(records, batch_size: int, epochs: int=1, shuffle=False):
    """Train and test."""
    if shuffle:
        dataset = tf.data.TFRecordDataset(records).repeat(count=epochs) \
            .shuffle(buffer_size=1000).batch(batch_size)
    else:
        dataset = tf.data.TFRecordDataset(records).repeat(count=epochs) \
            .batch(batch_size)
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

    return x_ord, x_ord_mask, x_cat, x_cat_mask, y


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

# def predict(model, records_test):

#     model_file = tf.train.latest_checkpoint(model)
#     print("Loading model: {}".format(model_file))

#     graph = tf.Graph()
#     with graph.as_default():
#         sess = tf.Session(config=predict_config)
#         with sess.as_default():
#             saver = tf.train.import_meta_graph("{}.meta".format(model_file))
#             saver.restore(sess, model_file)

#             # Restore place holders and prediction network
#             Xo = graph.get_operation_by_name("Inputs/Xo").outputs[0]
#             Xom = graph.get_operation_by_name("Inputs/Xom").outputs[0]
#             Xc = graph.get_operation_by_name("Inputs/Xc").outputs[0]
#             Xcm = graph.get_operation_by_name("Inputs/Xcm").outputs[0]
#             placeholders = [Xo, Xom, Xc, Xcm]

#             phi = graph.get_operation_by_name("Deepnet/nnet").outputs[0]
#             # TODO plus noise

#             datgen = extract_masks_query(data_test)
#             for i, d in enumerate(datgen):
#                 log.info("predicting batch {}".format(i))
#                 fd = dict(zip(placeholders, d[:4]))
#                 y_samples = ab.predict_samples(phi, fd, psamps, sess)
#                 Ey = y_samples.mean(axis=0)
#                 Sf = y_samples.std(axis=0)
#                 yield (Ey, Sf, *d[4:])


def train_test(records_train, records_test,
               train_metadata, test_metadata, name):

    train_dataset = dataset(records_train, batch_size, epochs=epochs,
                            shuffle=True)
    test_dataset = dataset(records_test, batch_size, epochs=1, shuffle=False)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xof, Xomf, Xcf, Xcmf, Y = decode(iterator, train_metadata)

    # lenscale = np.ones((Xof.shape[1], 1), dtype=np.float32) * 10.
    lenscale = 10.

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

        for i in count():
            log.info("Training round {} with {} epochs.".format(i, epochs))
            try:

                sess.run(train_init_op)
                try:
                    while not sess.should_stop():
                        sess.run(train)
                except tf.errors.OutOfRangeError:
                    log.info("Input queues have been exhausted!")

                EY = {}
                Ys = {}
                for _ in range(psamps):
                    sess.run(test_init_op)
                    j = 0
                    try:
                        while not sess.should_stop():
                            y, Eysamps = sess.run([Y, phi])
                            if j not in EY:
                                EY[j] = 0.
                                Ys[j] = y.squeeze()
                            EY[j] += Eysamps.mean(axis=0).squeeze()
                            j += 1
                    except tf.errors.OutOfRangeError:
                        pass
                EY = np.concatenate([EY[j] for j in range(psamps)]) / psamps
                Ys = np.concatenate([Ys[j] for j in range(psamps)])
                r2_score = rsquare(Ys, EY)
                rsquare_summary(r2_score, sess, i)
                log.info("Aboleth r2: {:.4f}".format(r2_score))

            except KeyboardInterrupt:
                log.info("Training ended.")
                break

    return checkpoint_dir


def rsquare(Y: np.ndarray, EY: np.ndarray) -> float:
    SS_ref = np.sum((Y - EY)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = float(1 - SS_ref / SS_tot)
    return R2

def rsquare_summary(r2_score, sess, step=None):
    # Get a summary writer for R-square
    summary_writer = sess._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag='r-square', simple_value=r2_score)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)
