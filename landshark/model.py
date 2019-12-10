"""Train/test with tfrecords."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import signal
from itertools import count
from typing import Any, Callable, Dict, Generator, List, NamedTuple, Optional

import numpy as np
import tensorflow as tf

from landshark.metadata import Training
from landshark.saver import BestScoreSaver
from landshark.serialise import deserialise

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


def train_data(records: List[str],
               metadata: Training,
               batch_size: int,
               epochs: int = 1,
               shuffle_buffer: int = 1000,
               take: Optional[int] = None,
               random_seed: Optional[int] = None
               ) -> Callable[[], tf.data.TFRecordDataset]:
    """Train dataset feeder."""
    take = -1 if take is None else take

    def f() -> tf.data.TFRecordDataset:
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .repeat(count=epochs) \
            .shuffle(buffer_size=shuffle_buffer, seed=random_seed) \
            .take(take) \
            .batch(batch_size) \
            .map(lambda x: deserialise(x, metadata))
        return dataset
    return f


def test_data(records: List[str],
              metadata: Training,
              batch_size: int
              ) -> Callable[[], tf.data.TFRecordDataset]:
    """Test and query dataset feeder."""
    def f() -> tf.data.TFRecordDataset:
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .batch(batch_size) \
            .map(lambda x: deserialise(x, metadata))
        return dataset
    return f


def predict_data(records: List[str],
                 metadata: Training,
                 batch_size: int
                 ) -> Callable[[], tf.data.TFRecordDataset]:
    """Test and query dataset feeder."""
    def f() -> tf.data.TFRecordDataset:
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .batch(batch_size) \
            .map(lambda x: deserialise(x, metadata, ignore_y=True))
        return dataset
    return f


def train_test(records_train: List[str],
               records_test: List[str],
               metadata: Training,
               directory: str,
               cf: Any,  # Module type
               params: TrainingConfig,
               iterations: Optional[int]
               ) -> None:
    """Model training and periodic hold-out testing."""
    saver = BestScoreSaver(directory)
    sess_config = tf.compat.v1.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})

    train_fn = train_data(records_train, metadata, params.batchsize,
                          params.epochs)
    test_fn = test_data(records_test, metadata, params.test_batchsize)

    run_config = tf.estimator.RunConfig(
        # tf_random_seed=params.seed,
        model_dir=directory,
        save_checkpoints_secs=300,
        session_config=sess_config,
        keep_checkpoint_max=1
    )

    estimator = tf.estimator.Estimator(
        model_fn=_model_wrapper,
        config=run_config,
        params={"metadata": metadata, "config": cf.model}
    )

    counter = range(iterations) if iterations else count()
    for i in counter:
        log.info("Training round {} with {} epochs.".format(i, params.epochs))
        try:
            estimator.train(input_fn=train_fn)
            eval_result = estimator.evaluate(input_fn=test_fn)
            _log_scores(eval_result)
            saver.save(eval_result)

        except KeyboardInterrupt:
            log.info("Training stopped on keyboard input")
            break


def predict(checkpoint_dir: str,
            cf: Any,  # Module type
            metadata: Training,
            records: List[str],
            params: QueryConfig
            ) -> Generator:
    """Load a model and predict results for record inputs."""
    sess_config = tf.compat.v1.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})
    predict_fn = predict_data(records, metadata, params.batchsize)
    run_config = tf.estimator.RunConfig(
        # tf_random_seed=params.seed,
        model_dir=checkpoint_dir,
        session_config=sess_config,
    )

    estimator = tf.estimator.Estimator(
        model_fn=_model_wrapper,
        config=run_config,
        params={"metadata": metadata, "config": cf.model}
    )
    it = estimator.predict(predict_fn, yield_single_examples=False)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return


#
# Private module utility functions
#

def _model_wrapper(features: Dict[str, tf.Tensor],
                   labels: tf.Tensor,
                   mode: tf.estimator.ModeKeys,
                   params: Dict[str, Any]
                   ) -> tf.estimator.EstimatorSpec:
    metadata = params["metadata"]
    model_fn = params["config"]

    con, con_mask, cat, cat_mask = None, None, None, None
    if "con" in features:
        con = features["con"]
        con_mask = features["con_mask"]
    if "cat" in features:
        cat = features["cat"]
        cat_mask = features["cat_mask"]
    result = model_fn(mode, con, con_mask, cat, cat_mask, labels,
                      features["indices"], features["coords"], metadata)
    return result


def _log_scores(scores: Dict[str, np.ndarray]) -> None:
    """Log testing scores."""
    scores_str = ", ".join(
        [f"{k}={v:.4f}" for k, v in scores.items() if k != "global_step"]
    )
    step_str = f"global_step={scores['global_step']}"
    logmsg = f"Evaluation scores ({step_str}): {scores_str}"
    log.info(logmsg)
