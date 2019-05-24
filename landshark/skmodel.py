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
from typing import Dict, Iterator, List, Optional

import numpy as np
from tqdm import tqdm

from landshark.metadata import Training
from landshark.tfread import query_data_it, xy_record_data

log = logging.getLogger(__name__)


def train_test(config_module: str,
               records_train: List[str],
               records_test: List[str],
               metadata: Training,
               model_dir: str,
               maxpoints: Optional[int],
               batchsize: int,
               random_seed: int
               ) -> None:
    """Train and test an sklean model."""
    log.info("Extracting and subsetting training data")
    npoints = maxpoints if maxpoints is not None else -1
    x_con, x_cat, indices, coords, y = xy_record_data(
        records=records_train,
        metadata=metadata,
        batchsize=batchsize,
        npoints=npoints,
        random_seed=random_seed,
    )
    xt_con, xt_cat, indicest, coordst, y_test = xy_record_data(
        records=records_test,
        metadata=metadata,
        batchsize=batchsize,
        shuffle=False,
    )

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
    """Run predictions on query data in batches using sklearn model."""
    model_path = os.path.join(modeldir, "skmodel.pickle")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    total_size = metadata.features.image.height * metadata.features.image.width
    with tqdm(total=total_size) as pbar:
        for xi in query_data_it(query_records, batch_size, metadata.features):
            x_con, x_cat, indices, coords = xi
            res = model.predict(*xi)
            pbar.update(indices.shape[0])
            yield res
