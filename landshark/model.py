"""Train/test with tfrecords."""
from collections import namedtuple
import signal
import logging
from itertools import count
import os.path


from tqdm import tqdm
import numpy as np
import tensorflow as tf
import aboleth as ab
from sklearn.metrics import accuracy_score, log_loss, r2_score

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


FDICT = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }

TrainingConfig = namedtuple("TrainingConfig",
                            ["epochs", "batchsize", "samples",
                             "test_batchsize", "test_samples", "use_gpu"])

QueryConfig = namedtuple("QueryConfig", ["batchsize", "samples",
                                         "percentiles", "use_gpu"])


def train_data(records: list, batch_size: int, epochs: int=1) \
        -> tf.data.TFRecordDataset:
    """Train dataset feeder."""
    dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
        .repeat(count=epochs) \
        .shuffle(buffer_size=1000) \
        .batch(batch_size)
    return dataset


def test_data(records: list, batch_size: int) -> tf.data.TFRecordDataset:
    """Test and query dataset feeder"""
    dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
        .batch(batch_size)
    return dataset


def decode(iterator, metadata):
    """Decode tf.record strings."""
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


def train_test(records_train, records_test, metadata, directory, cf, params):

    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})

    classification = metadata.target_dtype != np.float32

    # Placeholders
    _query_records = tf.placeholder_with_default(
        records_test, (None,), name="QueryRecords")
    _query_batchsize = tf.placeholder_with_default(
        tf.constant(params.test_batchsize, dtype=tf.int64),
        shape=tuple(), name="BatchSize")
    _samples = tf.placeholder_with_default(
        tf.constant(params.samples, dtype=tf.int32),
        shape=tuple(), name="NSamples")

    # Datasets
    train_dataset = train_data(records_train, params.batchsize, params.epochs)
    test_dataset = test_data(_query_records, _query_batchsize)
    with tf.name_scope("Sources"):
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes,
            shared_name="Iterator"
            )
    train_init_op = iterator.make_initializer(train_dataset, name="TrainInit")
    test_init_op = iterator.make_initializer(test_dataset, name="QueryInit")
    Xo, Xom, Xc, Xcm, Y = decode(iterator, metadata)

    # Model
    with tf.name_scope("Deepnet"):
        F, lkhood, loss, Y = cf.model(Xo, Xom, Xc, Xcm, Y, _samples, metadata)
        tf.identity(F, name="F_sample")
        tf.identity(ab.sample_mean(F), name="F_mean")
        tf.summary.scalar("loss", loss)

    # Training
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Testing / Querying
    with tf.name_scope("Test"):
        Y_samps = tf.identity(lkhood.sample(seed=next(ab.random.seedgen)),
                              name="Y_sample")
        test_fdict = {_samples: params.test_samples}

        if classification:
            prob = tf.reduce_mean(lkhood.probs, axis=0, name="prob")
            Ey = tf.argmax(prob, axis=1, name="Ey", output_type=tf.int32)
            acc, bacc, lp = 0.0, 0.0, -float("inf")
        else:
            logprob = tf.identity(lkhood.log_prob(Y), name="log_prob")
            Ey = tf.identity(ab.sample_mean(Y_samps), name="Y_mean")
            r2, lp = -float("inf"), float("inf")

    # Logging learning progress
    logger = tf.train.LoggingTensorHook(
        {"step": global_step, "loss": loss},
        every_n_secs=60)

    saver = tf.train.Saver()
    best_scores = [-1 * np.inf]

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
            log.info("Training round {} with {} epochs."
                     .format(i, params.epochs))
            try:

                # Train loop
                sess.run(train_init_op)
                step = train_loop(train, global_step, sess)

                # Test loop
                sess.run(test_init_op, feed_dict=test_fdict)
                if classification:
                    acc, bacc, lp = classify_test_loop(Y, Ey, prob, sess,
                                                       test_fdict, metadata,
                                                       step)
                    if lp > best_scores[0]:
                        best_scores = [lp, acc, bacc]
                        # Save the variables to disk.
                        rsess = sess._sess._sess._sess._sess
                        save_path = saver.save(
                            rsess, os.path.join(directory, "model_best.ckpt"))
                        log.info("New best model saved with lp: {}".format(lp))

                else:
                    r2, lp = regress_test_loop(Y, Ey, logprob, sess,
                                               test_fdict, metadata, step)

                    if r2 > best_score:
                        best_scores = [lp, r2]
                        # Save the variables to disk.
                        rsess = sess._sess._sess._sess._sess
                        save_path = saver.save(
                            rsess, os.path.join(directory, "model_best.ckpt"))
                        log.info("New best model saved with lp: {}".format(lp))


            except KeyboardInterrupt:
                log.info("Training stopped on keyboard input")
                if classification:
                    lp, acc, bacc = best_scores
                    log.info("Final acc: {:.5f}, Final bacc: {:.5f}, "
                             "Final lp: {:.5f}.".format(acc, bacc, lp))
                else:
                    lp, r2 = best_scores
                    log.info("Final r2: {}, Final mlp: {:.5f}.".format(r2, lp))
                break


def predict(model, metadata, records, params):
    """Load a model and predict results for record inputs."""
    total_size = metadata.image_spec.height * metadata.image_spec.width
    classification = metadata.target_dtype != np.float32

    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})
    model_file = tf.train.latest_checkpoint(model)
    print("Loading model: {}".format(model_file))
    tf.reset_default_graph()
    with tf.Session(config=sess_config) as sess:
        graph = tf.get_default_graph()
        save = tf.train.import_meta_graph("{}.meta".format(model_file))
        log.info("Restoring {}".format(model_file))
        save.restore(sess, model_file)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=sess_config)
        with sess.as_default():
            save = tf.train.import_meta_graph("{}.meta".format(model_file))
            save.restore(sess, model_file)

            # Restore place holders and prediction network
            _records = graph.get_operation_by_name("QueryRecords").outputs[0]
            _batchsize = graph.get_operation_by_name("BatchSize").outputs[0]
            _nsamples = graph.get_operation_by_name("NSamples").outputs[0]
            feed_dict = {_records: records, _batchsize: params.batchsize,
                         _nsamples: params.samples}

            # Restore prediction network
            it_op = graph.get_operation_by_name("QueryInit")

            if classification:
                Ey = graph.get_operation_by_name("Test/Ey").outputs[0]
                prob = graph.get_operation_by_name("Test/prob").outputs[0]
                eval_list = [Ey, prob]
            else:
                Ef = graph.get_operation_by_name("Deepnet/F_mean").outputs[0]
                F_samps = graph.get_operation_by_name("Deepnet/F_sample")\
                    .outputs[0]
                Per = ab.sample_percentiles(F_samps, params.percentiles)
                eval_list = [Ef, Per]

            # Initialise the dataset iterator
            sess.run(it_op, feed_dict=feed_dict)
            with tqdm(total=total_size) as pbar:
                while True:
                    try:
                        res = sess.run(eval_list, feed_dict=feed_dict)
                        pbar.update(res[0].shape[0])
                        yield res
                    except tf.errors.OutOfRangeError:
                        return


def train_loop(train, global_step, sess):
    """Train using an intialised Dataset iterator."""
    try:
        while not sess.should_stop():
            _, step = sess.run([train, global_step])
    except tf.errors.OutOfRangeError:
        log.info("Training epoch(s) complete.")

    return step


def classify_test_loop(Y, Ey, prob, sess, fdict, metadata, step):
    Eys = []
    Ys = []
    Ps = []
    try:
        while not sess.should_stop():
            y, p, ey = sess.run([Y, prob, Ey], feed_dict=fdict)
            Ys.append(y)
            Ps.append(p)
            Eys.append(ey)
    except tf.errors.OutOfRangeError:
        log.info("Testing epoch complete.")
    Ys = np.vstack(Ys)[:, 0]
    Ps = np.vstack(Ps)
    Ey = np.hstack(Eys)
    nlabels = len(metadata.target_map[0])
    labels = np.arange(nlabels)
    counts = np.bincount(Ys, minlength=nlabels)
    weights = np.zeros_like(counts, dtype=float)
    weights[counts != 0] = 1. / counts[counts != 0].astype(float)
    sample_weights = weights[Ys]
    acc = accuracy_score(Ys, Ey)
    bacc = accuracy_score(Ys, Ey, sample_weight=sample_weights)
    lp = -1 * log_loss(Ys, Ps, labels=labels)
    acc_summary(acc, sess, step)
    bacc_summary(bacc, sess, step)
    logloss_summary(lp, sess, step)
    log.info("Aboleth acc: {:.5f}, bacc: {:.5f}, lp: {:.5f}"
             .format(acc, bacc, lp))
    return acc, bacc, lp


def regress_test_loop(Y, Ey, logprob, sess, fdict, metadata, step):
    Ys, EYs, LP = [], [], []
    try:
        while not sess.should_stop():
            y, ey, lp = sess.run([Y, Ey, logprob], feed_dict=fdict)
            Ys.append(y)
            EYs.append(ey)
            LP.append(lp)
    except tf.errors.OutOfRangeError:
        log.info("Testing epoch complete.")
        pass
    Ys = np.vstack(Ys)
    EYs = np.vstack(EYs)
    lp = np.concatenate(LP, axis=1).mean()
    r2 = r2_score(Ys, EYs, multioutput="raw_values")
    rsquare_summary(r2, sess, metadata.target_labels, step)
    logprob_summary(lp, sess, step)
    log.info("Aboleth r2: {}, mlp: {:.5f}" .format(r2, lp))
    return r2, lp


def rsquare_summary(r2, session, labels, step=None):
    # Get a summary writer for R-square
    summary_writer = session._hooks[1]._summary_writer
    sum_val = [tf.Summary.Value(tag="r-square-{}".format(l), simple_value=r)
               for l, r in zip(labels, r2)]
    score_sum = tf.Summary(value=sum_val)
    summary_writer.add_summary(score_sum, step)


def logprob_summary(logprob, session, step=None):
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag="mean log prob", simple_value=logprob)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def logloss_summary(logloss, session, step=None):
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag="log loss", simple_value=logloss)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def acc_summary(acc, session, step=None):
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag="accuracy", simple_value=acc)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)

def bacc_summary(bacc, session, step=None):
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag="balanced accuracy", simple_value=bacc)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def patch_slices(metadata):
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    dim = npatch * metadata.nfeatures_cat
    begin = range(0, dim, npatch)
    end = range(npatch, dim + npatch, npatch)
    slices = [slice(b, e) for b, e in zip(begin, end)]
    return slices
