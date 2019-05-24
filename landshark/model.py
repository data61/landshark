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
import os
import signal
import sys
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from landshark.metadata import FeatureSet, Training
from landshark.saver import BestScoreSaver
from landshark.tfread import dataset_fn, get_query_meta, get_training_meta

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


def _load_config(module_name: str, path: str) -> None:
    # Load the model
    modspec = spec_from_file_location(module_name, path)
    cf = module_from_spec(modspec)
    if not modspec.loader:
        raise RuntimeError("Could not load configuration module")
    modspec.loader.exec_module(cf)  # type: ignore
    # needed for pickling??
    sys.modules[module_name] = cf


def load_model(config_file: str) -> str:
    module_name = "userconfig"
    _load_config(module_name, config_file)
    return module_name


def setup_training(
    config: str,
    directory: str
) -> Tuple[Training, List[str], List[str], str, str]:
    """Get metadata and records needed to train model."""
    metadata, training_records, testing_records = get_training_meta(directory)

    # Write the metadata
    name = os.path.basename(config).rsplit(".")[0] + \
        "_model_{}of{}".format(metadata.testfold, metadata.nfolds)
    model_dir = os.path.join(os.getcwd(), name)
    try:
        os.makedirs(model_dir)
    except FileExistsError:
        pass
    metadata.save(model_dir)

    # Load the model
    module_name = load_model(config)

    return metadata, training_records, testing_records, model_dir, module_name


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
    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})

    train_fn = dataset_fn(records_train, params.batchsize, metadata.features,
                          metadata.targets, params.epochs, shuffle=True)
    test_fn = dataset_fn(records_test, params.test_batchsize,
                         metadata.features, metadata.targets)

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


def setup_query(
    config: str,
    querydir: str,
    checkpoint: str,
) -> Tuple[Training, FeatureSet, List[str], int, int, str]:
    """Get metadata and records needed to make predictions."""
    query_meta, query_records, strip, nstrip = get_query_meta(querydir)
    train_meta = Training.load(checkpoint)
    module_name = load_model(config)
    return train_meta, query_meta, query_records, strip, nstrip, module_name


def predict(checkpoint_dir: str,
            cf: Any,  # Module type
            metadata: Training,
            records: List[str],
            params: QueryConfig
            ) -> Generator:
    """Load a model and predict results for record inputs."""
    sess_config = tf.ConfigProto(device_count={"GPU": int(params.use_gpu)},
                                 gpu_options={"allow_growth": True})
    predict_fn = dataset_fn(records, params.batchsize, metadata.features)
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
    height = metadata.features.image.height
    width = metadata.features.image.width
    total_size = (height * width) // params.batchsize
    with tqdm(total=total_size) as pbar:
        while True:
            try:
                yield next(it)
                pbar.update()
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
