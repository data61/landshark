"""Train/test with tfrecords."""
import signal
import logging
import os.path
from typing import List, Any, Generator
from itertools import count
from collections import namedtuple

import numpy as np
import tensorflow as tf
import aboleth as ab
from landshark.metadata import TrainingMetadata
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, r2_score

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


#
# Module constants and types
#

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


#
# Main functionality
#

def train_test(records_train: List[str],
               records_test: List[str],
               metadata: TrainingMetadata,
               directory: str,
               cf: Any,  # Module type
               params: TrainingConfig) -> None:
    """Model training and periodic hold-out testing."""
    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})

    classification = metadata.target_dtype != np.float32

    # Placeholders
    _query_records = tf.placeholder_with_default(
        records_test, (None,), name="QueryRecords")
    _query_batchsize = tf.placeholder_with_default(
        tf.constant(params.test_batchsize, dtype=tf.int64), shape=(),
        name="BatchSize")
    _samples = tf.placeholder_with_default(
        tf.constant(params.samples, dtype=tf.int32), shape=(), name="NSamples")

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
        else:
            logprob = tf.identity(lkhood.log_prob(Y), name="log_prob")
            Ey = tf.identity(ab.sample_mean(Y_samps), name="Y_mean")

    # Logging and saving learning progress
    logger = tf.train.LoggingTensorHook(
        {"step": global_step, "loss": loss},
        every_n_secs=60)
    saver = BestScoreSaver(directory)

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=sess_config,
            checkpoint_dir=directory,
            scaffold=tf.train.Scaffold(local_init_op=train_init_op),
            save_summaries_steps=None,
            save_checkpoint_secs=None,  # We will save model manually
            save_summaries_secs=20,
            log_step_count_steps=6000,
            hooks=[logger]
            ) as sess:

        saver.attach_session(sess._sess._sess._sess._sess)

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
                    *scores, lp = classify_test_loop(Y, Ey, prob, sess,
                                                     test_fdict, metadata,
                                                     step)
                else:
                    *scores, lp = regress_test_loop(Y, Ey, logprob, sess,
                                                    test_fdict, metadata, step)
                saver.save(lp, *scores)

            except KeyboardInterrupt:
                log.info("Training stopped on keyboard input")
                if classification:
                    lp, acc, bacc = saver.best_scores
                    log.info("Final acc: {:.5f}, Final bacc: {:.5f}, "
                             "Final lp: {:.5f}.".format(acc, bacc, lp))
                else:
                    lp, r2 = saver.best_scores
                    log.info("Final r2: {}, Final mlp: {:.5f}.".format(r2, lp))
                break


def predict(model: str,
            metadata: TrainingMetadata,
            records: List[str],
            params: QueryConfig) -> Generator:
    """Load a model and predict results for record inputs."""
    total_size = metadata.image_spec.height * metadata.image_spec.width
    classification = metadata.target_dtype != np.float32

    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})
    model_file = tf.train.latest_checkpoint(model)

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
            _records = load_op(graph, "QueryRecords")
            _batchsize = load_op(graph, "BatchSize")
            _nsamples = load_op(graph, "NSamples")

            if classification:
                Ey = load_op(graph, "Test/Ey")
                prob = load_op(graph, "Test/prob")
                eval_list = [Ey, prob]
            else:
                Ef = load_op(graph, "Deepnet/F_mean")
                F_samps = load_op(graph, "Deepnet/F_sample")
                Per = ab.sample_percentiles(F_samps, params.percentiles)
                eval_list = [Ef, Per]

            # Initialise the dataset iterator
            feed_dict = {_records: records, _batchsize: params.batchsize,
                         _nsamples: params.samples}
            it_op = graph.get_operation_by_name("QueryInit")
            sess.run(it_op, feed_dict=feed_dict)

            # Get a single set of samples from the model
            res = fix_samples(graph, sess, eval_list, feed_dict)

            with tqdm(total=total_size) as pbar:
                # Yeild prediction result from fixing samples
                yield res
                pbar.update(res[0].shape[0])

                # Continue obtaining predictions
                while True:
                    try:
                        res = sess.run(eval_list, feed_dict=feed_dict)
                        yield res
                        pbar.update(res[0].shape[0])
                    except tf.errors.OutOfRangeError:
                        return


#
# Module utility functions
#

def fix_samples(graph, sess, eval_list, feed_dict):
    """Fix the samples in an Aboleth graph for prediction.

    This also requires one evaluation of the graph, so the result is returned.
    """
    # Get the stochastic parameters from the graph (and the eval_list)
    params = graph.get_collection("SampleTensors")
    res = sess.run(eval_list + list(params), feed_dict=feed_dict)

    # Update the feed dict with these samples
    neval = len(eval_list)
    sample_feed_dict = dict(zip(params, res[neval:]))
    feed_dict.update(sample_feed_dict)

    return res[0:neval]


def train_data(records: List[str], batch_size: int, epochs: int=1) \
        -> tf.data.TFRecordDataset:
    """Train dataset feeder."""
    dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
        .repeat(count=epochs) \
        .shuffle(buffer_size=1000) \
        .batch(batch_size)
    return dataset


def test_data(records: List[str], batch_size: int) -> tf.data.TFRecordDataset:
    """Test and query dataset feeder."""
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


class BestScoreSaver:
    # TODO - see if we can make the "best score" persist between runs?

    def __init__(self, directory):
        self.path = os.path.join(directory, "model_best.ckpt")
        self.best_scores = [-1 * np.inf]
        self.saver = tf.train.Saver()

    def attach_session(self, session):
        self.sess = session

    def save(self, score, *other_scores):
        if score > self.best_scores[0]:
            self.best_scores = [score] + list(other_scores)
            self.saver.save(self.sess, self.path)
            log.info("New best model saved with score: {}".format(score))


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
    scalar_summary(acc, sess, "accuracy", step)
    scalar_summary(bacc, sess, "balanced accuracy", step)
    scalar_summary(lp, sess, "log probability", step)
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
    vector_summary(r2, sess, "r-square", metadata.target_labels, step)
    scalar_summary(lp, sess, "mean log prob", step)
    log.info("Aboleth r2: {}, mlp: {:.5f}" .format(r2, lp))
    return r2, lp


def scalar_summary(scalar, session, tag, step=None):
    """Add and update a summary scalar to TensorBoard."""
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag=tag, simple_value=scalar)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def vector_summary(vector, session, tag, labels, step=None):
    """Add and update a summary vector (list of scalars) to TensorBoard."""
    # Get a summary writer for R-square
    summary_writer = session._hooks[1]._summary_writer
    sum_val = [tf.Summary.Value(tag="{}-{}".format(tag, l), simple_value=v)
               for l, v in zip(labels, vector)]
    score_sum = tf.Summary(value=sum_val)
    summary_writer.add_summary(score_sum, step)


def patch_slices(metadata):
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    dim = npatch * metadata.nfeatures_cat
    begin = range(0, dim, npatch)
    end = range(npatch, dim + npatch, npatch)
    slices = [slice(b, e) for b, e in zip(begin, end)]
    return slices


def load_op(graph: tf.Graph, name: str) -> tf.Tensor:
    """Load an operation/tensor from a graph."""
    tensor = graph.get_operation_by_name(name).outputs[0]
    return tensor
