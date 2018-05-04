"""Train/test with tfrecords."""
import signal
import logging
import json
import os.path
from typing import List, Any, Generator, Optional, Dict, Union, \
    Iterable, Tuple, NamedTuple
from itertools import count

import numpy as np
import tensorflow as tf
import aboleth as ab
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, r2_score, \
    confusion_matrix

from landshark.basetypes import ClassificationPrediction, RegressionPrediction
from landshark.metadata import TrainingMetadata, CategoricalMetadata
from landshark.serialise import deserialise

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


#
# Module constants and types
#


class TrainingConfig(NamedTuple):
    epochs: int
    batchsize: int
    samples: int
    test_batchsize: int
    test_samples: int
    use_gpu: bool
    learnrate: float


class QueryConfig(NamedTuple):
    batchsize: int
    samples: int
    percentiles: Tuple[float, float]
    use_gpu: bool


#
# Main functionality
#

def train_test(records_train: List[str],
               records_test: List[str],
               metadata: TrainingMetadata,
               directory: str,
               cf: Any,  # Module type
               params: TrainingConfig,
               iterations: Optional[int]) -> None:
    """Model training and periodic hold-out testing."""
    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})

    classification = isinstance(metadata.targets, CategoricalMetadata)

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
    Xo, Xom, Xc, Xcm, Y = deserialise(iterator, metadata)

    # Model
    with tf.name_scope("Deepnet"):
        F, lkhood, loss, Y = cf.model(Xo, Xom, Xc, Xcm, Y, _samples, metadata)
        tf.identity(F, name="F_sample")
        tf.identity(ab.sample_mean(F), name="F_mean")
        tf.summary.scalar("loss", loss)

    # Training
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learnrate)
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Testing / Querying
    with tf.name_scope("Test"):
        Y_samps = tf.identity(lkhood.sample(seed=next(ab.random.seedgen)),
                              name="Y_sample")
        test_fdict = {_samples: params.test_samples}

        if classification:
            prob, Ey = _decision(lkhood)
        else:
            logprob = tf.identity(lkhood.log_prob(Y), name="log_prob")
            Ey = tf.identity(ab.sample_mean(Y_samps), name="Y_mean")

    # Logging and saving learning progress
    logger = tf.train.LoggingTensorHook(
        {"step": global_step, "loss": loss},
        every_n_secs=60)
    saver = _BestScoreSaver(directory)

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

        counter = range(iterations) if iterations else count()
        for i in counter:
            log.info("Training round {} with {} epochs."
                     .format(i, params.epochs))
            try:

                # Train loop
                sess.run(train_init_op)
                step = _train_loop(train, global_step, sess)

                # Test loop
                sess.run(test_init_op, feed_dict=test_fdict)
                if classification:
                    scores = _classify_test_loop(Y, Ey, prob, sess, test_fdict,
                                                 metadata, step)
                else:
                    scores = _regress_test_loop(Y, Ey, logprob, sess,
                                                test_fdict, metadata, step)
                saver.save(scores)
                _log_scores(scores, "Aboleth ")
            except KeyboardInterrupt:
                log.info("Training stopped on keyboard input")
                break


def predict(model: str,
            metadata: TrainingMetadata,
            records: List[str],
            params: QueryConfig) -> Generator:
    """Load a model and predict results for record inputs."""
    classification = isinstance(metadata.targets, CategoricalMetadata)

    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})
    model_file = tf.train.latest_checkpoint(model)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=sess_config)
        with sess.as_default():
            log.info("Restoring {}".format(model_file))
            save = tf.train.import_meta_graph("{}.meta".format(model_file))
            save.restore(sess, model_file)

            # Restore place holders and prediction network
            _records = _load_op(graph, "QueryRecords")
            _batchsize = _load_op(graph, "BatchSize")
            _nsamples = _load_op(graph, "NSamples")

            if classification:
                Ey = _load_op(graph, "Test/Ey")
                prob = _load_op(graph, "Test/prob")
                eval_list = [Ey, prob]
                to_obj: type = ClassificationPrediction
            else:
                Ef = _load_op(graph, "Deepnet/F_mean")
                F_samps = _load_op(graph, "Deepnet/F_sample")
                Per = ab.sample_percentiles(F_samps, params.percentiles)
                eval_list = [Ef, Per]
                to_obj = RegressionPrediction

            # Initialise the dataset iterator
            feed_dict = {_records: records, _batchsize: params.batchsize,
                         _nsamples: params.samples}
            it_op = graph.get_operation_by_name("QueryInit")
            sess.run(it_op, feed_dict=feed_dict)

            # Get a single set of samples from the model
            res = _fix_samples(graph, sess, eval_list, feed_dict)
            res_obj = to_obj(*res)

            total_size = (metadata.features.image.height *
                          metadata.features.image.width) // params.batchsize
            with tqdm(total=total_size) as pbar:
                # Yield prediction result from fixing samples
                yield res_obj
                pbar.update()

                # Continue obtaining predictions
                while True:
                    try:
                        res = to_obj(*sess.run(eval_list, feed_dict=feed_dict))
                        res_obj = to_obj(*res)
                        yield res_obj
                        pbar.update()
                    except tf.errors.OutOfRangeError:
                        return


def patch_slices(metadata: TrainingMetadata) -> List[slice]:
    """Get slices into the covariates corresponding to patches."""
    assert metadata.features.categorical
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    dim = npatch * metadata.features.categorical.D
    begin = range(0, dim, npatch)
    end = range(npatch, dim + npatch, npatch)
    slices = [slice(b, e) for b, e in zip(begin, end)]
    return slices

def patch_categories(metadata: TrainingMetadata) -> List[int]:
    """Get the number of categories including the extra patch columns"""
    assert metadata.features.categorical
    bmul = (2 * metadata.halfwidth + 1) ** 2
    ncats_nested = [[k] * bmul for k in
                    metadata.features.categorical.ncategories]
    ncategories_patched = [e for l in ncats_nested for e in l]
    return ncategories_patched

def train_data(records: List[str], batch_size: int, epochs: int=1,
               random_seed: Optional[int]=None) -> tf.data.TFRecordDataset:
    """Train dataset feeder."""
    dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
        .repeat(count=epochs) \
        .shuffle(buffer_size=1000, seed=random_seed) \
        .batch(batch_size)
    return dataset


def test_data(records: List[str], batch_size: int) -> tf.data.TFRecordDataset:
    """Test and query dataset feeder."""
    dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
        .batch(batch_size)
    return dataset


def sample_weights_labels(metadata: TrainingMetadata, Ys: np.array) -> \
        Tuple[np.array, np.array]:
    """Calculate the samples weights and labels for classification."""
    assert metadata.target_map is not None
    nlabels = len(metadata.target_map[0])
    labels = np.arange(nlabels)
    counts = np.bincount(Ys, minlength=nlabels)
    weights = np.zeros_like(counts, dtype=float)
    weights[counts != 0] = 1. / counts[counts != 0].astype(float)
    sample_weights = weights[Ys]
    return sample_weights, labels


#
# Private module utility functions
#

def _log_scores(scores: dict, initial_message: str="Aboleth ") -> None:
    """Log testing scores."""
    logmsg = str(initial_message)
    for k, v in scores.items():
        logmsg += "{}: {} ".format(k, v)
    log.info(logmsg)


def _fix_samples(graph: tf.Graph, sess: tf.Session, eval_list: List[tf.Tensor],
                 feed_dict: dict) -> Any:
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


def _decision(lkhood: tf.distributions.Distribution,
              binary_threshold: float=0.5) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get a decision from a binary or multiclass classifier."""
    prob = tf.reduce_mean(lkhood.probs, axis=0, name="prob")
    # Multiclass
    if prob.shape[1] > 1:
        Ey = tf.argmax(prob, axis=1, name="Ey", output_type=tf.int32)
    # Binary
    else:
        Ey = tf.squeeze(prob > binary_threshold, name="Ey")
    return prob, Ey


class _BestScoreSaver:
    """Saver for only saving the best model based on held out score.

    This now persists between runs by keeping a JSON file in the model
    directory.
    """

    def __init__(self, directory: str, score_name: str="lp") -> None:
        """Saver initialiser."""
        self.model_path = os.path.join(directory, "model_best.ckpt")
        self.score_path = os.path.join(directory, "model_best.json")
        self.score_name = score_name
        if os.path.exists(self.score_path):
            with open(self.score_path, "r") as f:
                self.best_scores = json.load(f)
        else:
            self.best_scores = {score_name: -1 * np.inf}
        self.saver = tf.train.Saver()

    def attach_session(self, session: tf.Session) -> None:
        """Attach a session to save."""
        self.sess = session

    def save(self, scores: dict) -> None:
        """Save the session *only* if the best score is exceeded."""
        if self.score_name not in scores:
            raise ValueError("score_name has to be in dictionary of scores!")
        if scores[self.score_name] > self.best_scores[self.score_name]:
            self.best_scores = scores
            self.saver.save(self.sess, self.model_path)
            with open(self.score_path, "w") as f:
                json.dump(self.best_scores, f)
            log.info("New best model saved with score: {}"
                     .format(self.best_scores[self.score_name]))


def _train_loop(train: tf.Tensor, global_step: tf.Tensor, sess: tf.Session)\
        -> Any:
    """Train using an intialised Dataset iterator."""
    step = 0
    try:
        while not sess.should_stop():
            _, step = sess.run([train, global_step])
    except tf.errors.OutOfRangeError:
        log.info("Training epoch(s) complete.")

    return step


def _classify_test_loop(Y: tf.Tensor, Ey: tf.Tensor, prob: tf.Tensor,
                        sess: tf.Session, fdict: dict,
                        metadata: TrainingMetadata, step: int) \
        -> Dict[str, Union[List[float], float]]:
    """Test the trained classifier on held out data."""
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
    sample_weights, labels = sample_weights_labels(metadata, Ys)
    acc = accuracy_score(Ys, Ey)
    bacc = accuracy_score(Ys, Ey, sample_weight=sample_weights)
    conf = confusion_matrix(Ys, Ey)
    lp = float(-1 * log_loss(Ys, Ps, labels=labels))
    _scalar_summary(acc, sess, "accuracy", step)
    _scalar_summary(bacc, sess, "balanced accuracy", step)
    _scalar_summary(lp, sess, "log probability", step)
    scores = {"acc": acc, "bacc": bacc, "lp": lp, "confmat": conf.tolist()}
    return scores


def _regress_test_loop(Y: tf.Tensor, Ey: tf.Tensor, logprob: tf.Tensor,
                       sess: tf.Session, fdict: dict,
                       metadata: TrainingMetadata, step: int) \
        -> Dict[str, Union[List[float], float]]:
    """Test the trained regressor on held out data."""
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
    lp = float(np.concatenate(LP, axis=1).mean())
    r2 = list(r2_score(Ys, EYs, multioutput="raw_values"))
    _vector_summary(r2, sess, "r-square", metadata.targets.labels, step)
    _scalar_summary(lp, sess, "mean log prob", step)
    scores = {"r2": r2, "lp": lp}
    return scores


def _scalar_summary(scalar: Union[int, bool, float], session: tf.Session,
                    tag: str, step: Optional[int]=None) -> None:
    """Add and update a summary scalar to TensorBoard."""
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag=tag, simple_value=scalar)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


def _vector_summary(vector: Iterable, session: tf.Session, tag: str,
                    labels: Iterable, step: Optional[int]=None) -> None:
    """Add and update a summary vector (list of scalars) to TensorBoard."""
    # Get a summary writer for R-square
    summary_writer = session._hooks[1]._summary_writer
    sum_val = [tf.Summary.Value(tag="{}-{}".format(tag, l), simple_value=v)
               for l, v in zip(labels, vector)]
    score_sum = tf.Summary(value=sum_val)
    summary_writer.add_summary(score_sum, step)


def _load_op(graph: tf.Graph, name: str) -> tf.Tensor:
    """Load an operation/tensor from a graph."""
    tensor = graph.get_operation_by_name(name).outputs[0]
    return tensor
