"""Train/test with tfrecords."""

import json
import logging
import os.path
from glob import glob
import shutil
import signal
from itertools import count
from typing import (Any, Dict, Generator, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)

import aboleth as ab
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             r2_score)
from tqdm import tqdm

from landshark.basetypes import ClassificationPrediction, RegressionPrediction
from landshark.metadata import CategoricalMetadata, TrainingMetadata
from landshark.serialise import deserialise

import aboleth as ab

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


#
# Module constants and types
#


class TrainingConfig(NamedTuple):
    epochs: int
    batchsize: int
    test_batchsize: int
    use_gpu: bool


class QueryConfig(NamedTuple):
    batchsize: int
    use_gpu: bool


def train_data(records: List[str], metadata: TrainingMetadata,
               batch_size: int, epochs: int=1, shuffle_buffer: int=1000,
               random_seed: Optional[int]=None) \
        -> tf.data.TFRecordDataset:
    """Train dataset feeder."""
    def f():
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .repeat(count=epochs) \
            .shuffle(buffer_size=shuffle_buffer, seed=random_seed) \
            .batch(batch_size)
        raw_data = dataset.make_one_shot_iterator().get_next()
        x, y = deserialise(raw_data, metadata)
        return x, y
    return f


def test_data(records: List[str], metadata: TrainingMetadata,
                      batch_size: int) -> tf.data.TFRecordDataset:
    """Test and query dataset feeder."""
    def f():
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .batch(batch_size)
        raw_data = dataset.make_one_shot_iterator().get_next()
        x, y = deserialise(raw_data, metadata)
        return x, y
    return f

def predict_data(records: List[str], metadata: TrainingMetadata,
                      batch_size: int) -> tf.data.TFRecordDataset:
    """Test and query dataset feeder."""
    def f():
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .batch(batch_size)
        raw_data = dataset.make_one_shot_iterator().get_next()
        x, _ = deserialise(raw_data, metadata)
        return x
    return f


class _BestScoreSaver:
    """Saver for only saving the best model based on held out score.

    This now persists between runs by keeping a JSON file in the model
    directory.
    """

    def __init__(self, directory: str) -> None:
        """Saver initialiser."""
        self.directory = directory

    def _init_dir(self, score_path) -> None:
        if not os.path.exists(score_path):
            os.mkdir(score_path)
            shutil.copy2(os.path.join(self.directory, "METADATA.bin"),
                         score_path)

    def _to_64bit(self, scores: Dict[str, np.ndarray]) -> None:
        # convert scores to 64bit
        for k, v in scores.items():
            if v.dtype == np.float32:
                scores[k] = v.astype(np.float64)
            if v.dtype == np.int32:
                scores[k] = v.astype(np.int64)


    def _should_overwrite(self, s: str, score: np.ndarray,
                          score_path: str) -> bool:
        score_file = os.path.join(score_path, "model_best.json")
        overwrite = True
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                best_scores = json.load(f)
            if s == "loss":
                if best_scores[s] < score:
                    overwrite = False
            else:
                if best_scores[s] > score:
                    overwrite = False
        return overwrite


    def _write_score(self, scores: Dict[str, np.ndarray],
                     score_path: str, global_step: int) -> None:
        score_file = os.path.join(score_path, "model_best.json")
        with open(score_file, 'w') as f:
            json.dump(scores, f)
        checkpoint_files = glob(os.path.join(self.directory,
                                "model.ckpt-{}*".format(global_step)))
        deleting_files = glob(os.path.join(score_path, "model.ckpt-*"))
        for d in deleting_files:
            os.remove(d)
        for c in checkpoint_files:
            shutil.copy2(c, score_path)

    def save(self, scores: dict) -> None:
        global_step = scores.pop("global_step")
        # Create directories if they don't exist
        for s in scores.keys():
            score_path = self.directory + "_best_{}".format(s)
            self._init_dir(score_path)
            if self._should_overwrite(s, scores[s], score_path):
                log.info("Found model with new best {} score: overwriting".
                         format(s))
                self._write_score(s, score_path, global_step)


def train_test(records_train: List[str],
               records_test: List[str],
               metadata: TrainingMetadata,
               directory: str,
               cf: Any,  # Module type
               params: TrainingConfig,
               iterations: Optional[int]) -> None:
    """Model training and periodic hold-out testing."""


    saver = _BestScoreSaver(directory)
    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})

    train_fn = train_data(records_train, metadata,
                      params.batchsize, params.epochs)
    test_fn = test_data(records_test, metadata, params.test_batchsize)

    estimator = tf.estimator.Estimator(
        model_fn=cf.model,
        model_dir=directory,
        params={"metadata": metadata})

    for i in range(10):
        estimator.train(input_fn=train_fn, steps=5)
        eval_result = estimator.evaluate(input_fn=test_fn)
        saver.save(eval_result)
        print(eval_result)
    import IPython; IPython.embed(); import sys; sys.exit()


    ## Logging and saving learning progress
    #logger = tf.train.LoggingTensorHook(
    #    {"step": global_step, "loss": loss},
    #    every_n_secs=60)

    ## This is the main training "loop"
    #with tf.train.MonitoredTrainingSession(
    #        config=sess_config,
    #        checkpoint_dir=directory,
    #        scaffold=tf.train.Scaffold(local_init_op=train_init_op),
    #        save_summaries_steps=None,
    #        save_checkpoint_secs=None,  # We will save model manually
    #        save_summaries_secs=20,
    #        log_step_count_steps=6000,
    #        hooks=[logger]
    #        ) as sess:

    #    saver.attach_session(sess._sess._sess._sess._sess)

    #    counter = range(iterations) if iterations else count()
    #    for i in counter:
    #        log.info("Training round {} with {} epochs."
    #                 .format(i, params.epochs))
    #        try:

    #            # Train loop
    #            sess.run(train_init_op)
    #            step = _train_loop(train, global_step, sess)

    #            # Test loop
    #            sess.run(test_init_op, feed_dict=test_fdict)
    #            if classification:
    #                scores = _classify_test_loop(Y, Ey, prob, sess, test_fdict,
    #                                             metadata, step)
    #            else:
    #                scores = _regress_test_loop(Y, Ey, logprob, sess,
    #                                            test_fdict, metadata, step)
    #            saver.save(scores)
    #            _log_scores(scores, "Aboleth ")
    #        except KeyboardInterrupt:
    #            log.info("Training stopped on keyboard input")
    #            break

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
    """Get the number of categories including the extra patch columns."""
    assert metadata.features.categorical
    bmul = (2 * metadata.halfwidth + 1) ** 2
    ncats_nested = [[k] * bmul for k in
                    metadata.features.categorical.ncategories]
    ncategories_patched = [e for l in ncats_nested for e in l]
    return ncategories_patched



def sample_weights_labels(metadata: TrainingMetadata, Ys: np.array) -> \
        Tuple[np.array, np.array]:
    """Calculate the samples weights and labels for classification."""
    assert isinstance(metadata.targets, CategoricalMetadata)
    # Currently we only support single-task classification problems
    nlabels = metadata.targets.ncategories[0]
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
    if "confmat" in scores:
        del scores["confmat"]
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
    ax = 1 if len(prob.shape) > 1 else 0
    if prob.shape[ax] > 1:
        Ey = tf.argmax(prob, axis=ax, name="Ey", output_type=tf.int32)
    # Binary
    else:
        Ey = tf.squeeze(prob > binary_threshold, name="Ey")
    return prob, Ey


# class _BestScoreSaver:
#     """Saver for only saving the best model based on held out score.

#     This now persists between runs by keeping a JSON file in the model
#     directory.
#     """

#     def __init__(self, directory: str, score_name: str="lp") -> None:
#         """Saver initialiser."""
#         self.model_path = os.path.join(directory, "model_best.ckpt")
#         self.score_path = os.path.join(directory, "model_best.json")
#         self.score_name = score_name
#         if os.path.exists(self.score_path):
#             with open(self.score_path, "r") as f:
#                 self.best_scores = json.load(f)
#         else:
#             self.best_scores = {score_name: -1 * np.inf}
#         self.saver = tf.train.Saver()

#     def attach_session(self, session: tf.Session) -> None:
#         """Attach a session to save."""
#         self.sess = session

#     def save(self, scores: dict) -> None:
#         """Save the session *only* if the best score is exceeded."""
#         if self.score_name not in scores:
#             raise ValueError("score_name has to be in dictionary of scores!")
#         if scores[self.score_name] > self.best_scores[self.score_name]:
#             self.best_scores = scores
#             self.saver.save(self.sess, self.model_path)
#             with open(self.score_path, "w") as f:
#                 json.dump(self.best_scores, f)
#             log.info("New best model saved with score: {}"
#                      .format(self.best_scores[self.score_name]))


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
