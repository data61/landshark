"""Train/test with tfrecords."""
import signal
import logging
import os.path
import pickle
from itertools import count

import numpy as np
import tensorflow as tf
import aboleth as ab
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


FDICT = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }

BORING_QUANTILES = (
    tf.distributions.Bernoulli,
    tf.distributions.Categorical
    )


def train_data(records: list, batch_size: int, epochs: int=1) \
        -> tf.data.TFRecordDataset:
    """Train dataset feeder."""
    dataset = tf.data.TFRecordDataset(records,
                                      compression_type="ZLIB").repeat(count=epochs) \
        .shuffle(buffer_size=1000).batch(batch_size)
    return dataset


def test_data(records: list, batch_size: int, pred_samps: int=1) \
        -> tf.data.TFRecordDataset:
    """Train and test."""
    dataset = tf.data.TFRecordDataset(records,
                                      compression_type="ZLIB").batch(batch_size).interleave(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(pred_samps),
        cycle_length=1,
        block_length=pred_samps
        )
    return dataset


def decode(iterator, metadata):
    str_features = iterator.get_next()
    raw_features = tf.parse_example(str_features, features=FDICT)
    npatch = (2 * metadata.halfwidth + 1) ** 2
    y_type = tf.float32 if metadata.target_dtype == np.float32 \
        else tf.int32
    with tf.name_scope("Inputs"):
        x_ord = tf.decode_raw(raw_features["x_ord"], tf.float32)
        x_cat = tf.decode_raw(raw_features["x_cat"], tf.int32)
        x_ord_mask = tf.decode_raw(raw_features["x_ord_mask"], tf.uint8)
        x_cat_mask = tf.decode_raw(raw_features["x_cat_mask"], tf.uint8)
        x_ord_mask = tf.cast(x_ord_mask, tf.bool)
        x_cat_mask = tf.cast(x_cat_mask, tf.bool)
        y = tf.decode_raw(raw_features["y"], y_type)

        x_ord.set_shape((None, npatch * metadata.nfeatures_ord))
        x_ord_mask.set_shape((None, npatch * metadata.nfeatures_ord))
        x_cat.set_shape((None, npatch * metadata.nfeatures_cat))
        x_cat_mask.set_shape((None, npatch * metadata.nfeatures_cat))
        y.set_shape((None, metadata.ntargets))

    return x_ord, x_ord_mask, x_cat, x_cat_mask, y


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def predict(model, metadata, records, batch_size, pred_samples, lower, upper, use_gpu=False):

    sess_config = tf.ConfigProto(device_count={"GPU": int(use_gpu)},
                                 gpu_options={'allow_growth': True})
    model_file = tf.train.latest_checkpoint(model)
    print("Loading model: {}".format(model_file))

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=sess_config)
        with sess.as_default():
            # TODO AL reloads/rewrites the graph in memory from protobuf
            # See glabrezu
            save = tf.train.import_meta_graph("{}.meta".format(model_file))
            save.restore(sess, model_file)

            # Restore place holders and prediction network
            _records = graph.get_operation_by_name("QueryRecords").outputs[0]
            it_op = graph.get_operation_by_name("QueryInit")
            _batchsize = graph.get_operation_by_name("BatchSize").outputs[0]
            _predsamps = graph.get_operation_by_name("PredSamples").outputs[0]
            F = graph.get_operation_by_name("Test/Y_sample").outputs[0]
            feed_dict = {_records:records, _batchsize: batch_size,
                         _predsamps: pred_samples}
            sess.run(it_op, feed_dict=feed_dict)
            while True:
                try:
                    samples = []
                    for i in range(pred_samples):
                        samples.append(sess.run(F, feed_dict=feed_dict))
                    all_samples = np.concatenate(samples, axis=0)
                    Ey = all_samples.mean(axis=0)
                    Ly, Uy = np.percentile(all_samples, q=[lower, upper], axis=0) \
                    .astype(Ey.dtype)
                    yield Ey, Ly, Uy
                except tf.errors.OutOfRangeError:
                    return


def train_test(records_train, records_test, metadata, directory, batch_size, epochs,
               pred_samples, cf, use_gpu=False):

    sess_config = tf.ConfigProto(device_count={"GPU": int(use_gpu)},
                                 gpu_options={'allow_growth': True})
    train_dataset = train_data(records_train, batch_size, epochs)
    query_records = tf.placeholder_with_default(records_test, (None,),
                                                name="QueryRecords")
    _batchsize = tf.placeholder_with_default(tf.constant(batch_size, dtype=tf.int64), shape=tuple(),
                                             name="BatchSize")
    _predsamps = tf.placeholder_with_default(tf.constant(pred_samples, dtype=tf.int64), shape=tuple(),
                                             name="PredSamples")
    test_dataset = test_data(query_records, _batchsize, _predsamps)

    with tf.name_scope("Sources"):
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes,
            shared_name="Iterator"
            )
    train_init_op = iterator.make_initializer(train_dataset, name="TrainInit")
    test_init_op = iterator.make_initializer(test_dataset, name="QueryInit")

    Xo, Xom, Xc, Xcm, Y = decode(iterator, metadata)

    with tf.name_scope("Deepnet"):
        F, lkhood, loss = cf.model(Xo, Xom, Xc, Xcm, Y, metadata)
        tf.summary.scalar("loss", loss)

    # Set up the training graph
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Name some testing tensors for evaluation and prediction
    with tf.name_scope("Test"):
        F = tf.identity(F, name="F_sample")
        logprob = tf.identity(lkhood.log_prob(Y), name="log_prob")

        # Quantiles some distributions are trivial, so use the latent function
        if isinstance(lkhood, BORING_QUANTILES):
            tf.identity(F, name="Y_sample")
        else:
            tf.identity(lkhood.sample(seed=next(ab.random.seedgen)),
                        name="Y_sample")

    # Logging learning progress
    logger = tf.train.LoggingTensorHook(
        {"step": global_step, "loss": loss},
        every_n_secs=60
        )

    r2, lp = -float("inf"), float("inf")

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=sess_config,
            checkpoint_dir=directory,
            scaffold=tf.train.Scaffold(local_init_op=train_init_op),
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            log_step_count_steps=6000,
            hooks=[logger]
            ) as sess:

        for i in count():
            log.info("Training round {} with {} epochs.".format(i, epochs))
            try:

                # Train loop
                sess.run(train_init_op)
                step = train_loop(train, global_step, sess)

                # Test loop
                sess.run(test_init_op)
                Ys, EYs, lp = test_loop(Y, F, logprob, pred_samples, sess)

                # Score
                r2 = r2_score(Ys, EYs, multioutput='raw_values')
                rsquare_summary(r2, sess, metadata.target_labels, step)
                logprob_summary(lp, sess, step)
                log.info("Aboleth r2: {}, mlp: {:.5f}"
                         .format(r2, lp))

            except KeyboardInterrupt:
                log.info("Training stopped on keyboard input")
                log.info("Final r2: {}, Final mlp: {:.5f}.".format(r2, lp))
                break


def train_loop(train, global_step, sess):
    try:
        while not sess.should_stop():
            _, step = sess.run([train, global_step])
    except tf.errors.OutOfRangeError:
        log.info("Training epoch complete.")

    return step


def test_loop(Y, F, logprob, n_samples, sess):
    Ys, EYs, LP = [], [], []
    try:
        while not sess.should_stop():
            yaccum, lpaccum = 0., 0.
            for j in range(n_samples):
                y, ey, lp = sess.run([Y, F, logprob])
                yaccum += ey
                lpaccum += lp

            Ys.append(y)
            EYs.append(yaccum.mean(axis=0) / n_samples)
            LP.append(lpaccum.mean() / n_samples)
    except tf.errors.OutOfRangeError:
        log.info("Testing epoch complete.")
        pass

    Ys = np.vstack(Ys)
    EYs = np.vstack(EYs)
    LP = np.mean(LP)
    return Ys, EYs, LP


def rsquare_summary(r2, session, labels, step=None):
    # Get a summary writer for R-square
    summary_writer = session._hooks[1]._summary_writer
    sum_val = [tf.Summary.Value(tag='r-square-{}'.format(l), simple_value=r)
               for l, r in zip(labels, r2)]
    score_sum = tf.Summary(value=sum_val)
    summary_writer.add_summary(score_sum, step)


def logprob_summary(logprob, session, step=None):
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag='mean log prob', simple_value=logprob)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def patch_slices(metadata):
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    dim = npatch * metadata.nfeatures_cat
    begin = range(0, dim, npatch)
    end = range(npatch, dim + npatch, npatch)
    slices = [slice(b, e) for b, e in zip(begin, end)]
    return slices
