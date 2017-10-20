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
train_config = tf.ConfigProto(device_count={"GPU": 1})
predict_config = tf.ConfigProto(device_count={"GPU": 1})

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


def dataset_query(data, metadata, batch_size):

    npatch = (2 * metadata.halfwidth + 1) ** 2
    shapes = {
        "x_ord": npatch * metadata.x_ord,
        "x_ord_mask": npatch * metadata.x_ord,
        "x_cat": npatch * metadata.x_cat,
        "x_cat_mask": npatch * metadata.x_cat,
        # 1
        }
    types = {
        "x_ord": tf.float32,
        "x_ord_mask": tf.bool,
        "x_cat": tf.int32,
        "x_cat_mask": tf.bool
        }

    def slicer():
        for d in data:
            for xo, xc in zip(d.x_ord, d.x_cat):
                xo = xo.flatten()
                xc = xc.flatten()
                tslice = (xo.data, xo.mask, xc.data, xc.mask)
                yield tslice

    # Make the training data iterator
    dataset = tf.data.Dataset.from_generator(slicer, types, shapes) \
        .batch(batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    data_init_op = iterator.make_initializer(dataset)
    return data_init_op


def predict(model, data, metadata, batch_size):

    model_file = tf.train.latest_checkpoint(model)
    print("Loading model: {}".format(model_file))

    data_init_op = dataset_query(data, metadata, batch_size)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=predict_config)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(model_file))
            saver.restore(sess, model_file)
            phi = graph.get_operation_by_name("Deepnet/nnet").outputs[0]
#             # TODO plus noise

            sess.run(data_init_op)
            for i in count():
                if sess.should_stop():
                    break

                log.info("predicting batch {}".format(i))
                ys = sess.run(phi)
                Ey = ys.mean(axis=0)
                Sf = ys.std(axis=0)
                yield Ey, Sf


def train_test(records_train, records_test, train_metadata, test_metadata,
               name):

    train_dataset = dataset(records_train, batch_size, epochs=epochs,
                            shuffle=True)
    test_dataset = dataset(records_test, batch_size, epochs=1, shuffle=False)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xof, Xomf, Xcf, Xcmf, Y = decode(iterator, train_metadata)

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
    r2_score = None

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=train_config,
            checkpoint_dir=checkpoint_dir,
            scaffold=tf.train.Scaffold(local_init_op=train_init_op),
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            hooks=[logger]
            ) as sess:

        for i in count():
            log.info("Training round {} with {} epochs.".format(i, epochs))
            try:

                # Train loop
                try:
                    while not sess.should_stop():
                        sess.run(train)
                except tf.errors.OutOfRangeError:
                    log.info("Input queues have been exhausted!")

                # Test loop
                Ys, stats = {}, {}
                for _ in range(psamps):
                    sess.run(test_init_op)
                    for j, y, ey in prediction_gen(Y, phi, sess):
                        if j not in stats:
                            stats[j] = (None, None, None)
                            Ys[j] = y
                        stats[j] = incremental_stats(ey, *stats[j])

                # Scores
                r2_score = rsquare(Ys, stats)
                rsquare_summary(r2_score, sess, i)
                log.info("Aboleth r2: {:.4f}".format(r2_score))

                sess.run(train_init_op)

            except KeyboardInterrupt:
                log.info("Training ended, final R-square = {:.4f}."
                         .format(r2_score))
                break

    return checkpoint_dir


def prediction_gen(Y, phi, session):
    try:
        for i in count():
            if session.should_stop():
                break
            y, Eysamps = session.run([Y, phi])
            yield i, y, Eysamps
    except tf.errors.OutOfRangeError:
        pass


def incremental_stats(samples, mean=None, var=None, count=None):
    count_s = samples.shape[0]
    mean_s = samples.mean(axis=0)
    var_s = samples.var(axis=0)

    if (mean is None) or (var is None) or (count is None):
        return mean_s, var_s, count_s

    N = count_s + count
    delta = mean - mean_s
    ss_s = var_s * count_s
    ss = var * count

    mean = mean_s + delta * count / N
    var = (ss_s + ss + delta**2 * count * count_s / N) / N

    return mean, var, N


def rsquare(Y: dict, stats: dict) -> float:
    assert len(Y) == len(stats)
    Y = np.concatenate([Y[j] for j in range(len(Y))])
    EY = np.concatenate([stats[j][0] for j in range(len(stats))])

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
