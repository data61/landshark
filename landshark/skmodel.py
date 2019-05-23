"""Scikit Learn training and testing with tf records."""

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

import json
import logging
import os.path
import pickle
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from landshark.tfread import get_traintest_data, _make_mask, _split
from landshark.metadata import Training
from landshark.model import predict_data

log = logging.getLogger(__name__)


# TODO simplify now I'm no longer using the recursive dict structure


def _query_it(records_query: List[str],
              batch_size: int,
              metadata: Training
              ) -> Iterator[Dict[str, np.ndarray]]:

    total_size = metadata.features.image.height * metadata.features.image.width
    dataset = predict_data(records_query, metadata, batch_size)()
    X_tensor = dataset.make_one_shot_iterator().get_next()
    with tqdm(total=total_size) as pbar:
        with tf.Session() as sess:
            while True:
                try:
                    X = sess.run(X_tensor)
                    if "con" in X:
                        X["con"] = _make_mask(X["con"], X["con_mask"])
                    if "cat" in X:
                        X["cat"] = _make_mask(X["cat"], X["cat_mask"])
                    n = X["indices"].shape[0]
                    pbar.update(n)
                    yield X
                except tf.errors.OutOfRangeError:
                    break
            return


def train_test(config_module: str,
               records_train: List[str],
               records_test: List[str],
               metadata: Training,
               model_dir: str,
               maxpoints: Optional[int],
               batchsize: int,
               random_seed: int
               ) -> None:

    log.info("Extracting and subsetting training data")
    x, y, x_test, y_test = get_traintest_data(
        records_train, records_test, metadata, maxpoints, batchsize,
        random_seed
    )
    x_con, x_cat, indices, coords = _split(x)
    xt_con, xt_cat, indicest, coordst = _split(x_test)

    userconfig = __import__(config_module)

    log.info("Training model")
    model = userconfig.SKModel(metadata, random_seed=random_seed)

    model.train(x_con, x_cat, indices, coords, y)
    log.info("Evaluating test data")
    res = model.predict(xt_con, xt_cat, indicest, coordst)
    scores = model.test(y_test, res)
    log.info("Sklearn test metrics: {}".format(scores))

    log.info("Saving model to disk")
    model_path = os.path.join(model_dir, "skmodel.pickle")
    with open(model_path, "wb") as fb:
        pickle.dump(model, fb)

    score_path = os.path.join(model_dir, "skmodel.json")
    with open(score_path, "w") as f:
        json.dump(scores, f)


def predict(modeldir: str,
            metadata: Training,
            query_records: List[str],
            batch_size: int
            ) -> Iterator[Dict[str, np.ndarray]]:

    model_path = os.path.join(modeldir, "skmodel.pickle")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    for xi in _query_it(query_records, batch_size, metadata):
        x_con, x_cat, indices, coords = _split(xi)
        res = model.predict(x_con, x_cat, indices, coords)
        yield res
