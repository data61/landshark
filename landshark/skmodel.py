"""Scikit Learn training and testing with tf records."""

import json
import logging
import os.path
import pickle
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             r2_score)
from tqdm import tqdm

from landshark.basetypes import CategoricalType, ContinuousType
from landshark.metadata import CategoricalMetadata, TrainingMetadata
from landshark.model import train_data, test_data, predict_data
from landshark.serialise import deserialise

log = logging.getLogger(__name__)


def _make_mask(x: Dict[str, np.ndarray], label: str):
    if label in x:
        a = x.pop(label)
        m = x.pop(label + "_mask")
        ma = np.ma.MaskedArray(data=a, mask=m)
        x[label] = ma


def _extract(xt: Dict[str, tf.Tensor], yt: tf.Tensor, sess: tf.Session):

    x_list = []
    y_list = []
    try:
        while True:
            x, y = sess.run([xt, yt])
            x_list.append(x)
            y_list.append(y)
    except tf.errors.OutOfRangeError:
        pass

    y_full = np.concatenate(y_list, axis=0)
    x_full = {k: np.concatenate([di[k] for di in x_list], axis=0)
              for k,v in x_list[0].items() if v.shape[-1] != 0}

    _make_mask(x_full, "con")
    _make_mask(x_full, "cat")

    return x_full, y_full

def _get_data(records_train: List[str], records_test: List[str],
              metadata: TrainingMetadata, npoints: Optional[int],
              batch_size: int,
              random_seed: int) -> Tuple[Dict[str, np.ndarray], np.ndarray,
                                         Dict[str, np.ndarray], np.ndarray]:

    train_dataset = train_data(records_train, metadata, batch_size,
                      epochs=1, take=npoints, random_seed=random_seed)()
    X_tensor, Y_tensor = train_dataset.make_one_shot_iterator().get_next()
    test_dataset = test_data(records_test, metadata, batch_size)()
    Xt_tensor, Yt_tensor = test_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        X, Y = _extract(X_tensor, Y_tensor, sess)
        Xt, Yt = _extract(Xt_tensor, Yt_tensor, sess)
    return X, Y, Xt, Yt


def _query_it(records_query: List[str], batch_size: int,
              metadata: TrainingMetadata) \
        -> Iterator[Dict[str, np.ndarray]]:

    total_size = metadata.features.image.height * metadata.features.image.width
    dataset = predict_data(records_query, metadata, batch_size)()
    X_tensor = dataset.make_one_shot_iterator().get_next()
    with tqdm(total=total_size) as pbar:
        with tf.Session() as sess:
            while True:
                try:
                    X = sess.run(X_tensor)
                    X = {k: v for k, v in X.items() if v.shape[-1] > 0}
                    _make_mask(X, "con")
                    _make_mask(X, "cat")
                    n = X['indices'].shape[0]
                    pbar.update(n)
                    yield X
                except tf.errors.OutOfRangeError:
                    break
            return

def train_test(config_module: str, records_train: List[str],
               records_test: List[str], metadata: TrainingMetadata,
               model_dir: str, maxpoints: Optional[int], batchsize: int,
               random_seed: int) -> None:

    log.info("Extracting and subsetting training data")
    data_tuple = _get_data(records_train, records_test, metadata, maxpoints,
                           batchsize, random_seed)
    x, y, x_test, y_test = data_tuple

    userconfig = __import__(config_module)

    log.info("Training model")
    model = userconfig.SKModel(metadata, random_seed=random_seed)

    model.train(x, y)
    log.info("Evaluating test data")
    res = model.predict(x_test)
    scores = model.test(y_test, res)
    log.info("Sklearn test metrics: {}".format(scores))

    log.info("Saving model to disk")
    model_path = os.path.join(model_dir, "skmodel.pickle")
    with open(model_path, "wb") as fb:
        pickle.dump(model, fb)

    score_path = os.path.join(model_dir, "skmodel.json")
    with open(score_path, "w") as f:
        json.dump(scores, f)


def predict(modeldir: str, metadata: TrainingMetadata,
            query_records: List[str], batch_size: int) \
        -> Iterator[Dict[str, np.ndarray]]:

    model_path = os.path.join(modeldir, "skmodel.pickle")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    for xi in _query_it(query_records, batch_size, metadata):
        res = model.predict(xi)
        yield res
