"""Scikit Learn training and testing with tf records."""

import json
import logging
import os.path
import pickle
from typing import Iterator, List, Optional, Tuple, Dict, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from landshark.metadata import Training
from landshark.model import train_data, test_data, predict_data
from landshark.serialise import deserialise

log = logging.getLogger(__name__)



# TODO simplify now I'm no longer using the recursive dict structure


def _make_mask(x: Dict[str, np.ndarray],
               xm: Dict[str, np.ndarray]) -> Dict[str, np.ma.MaskedArray]:
    assert x.keys() == xm.keys()
    d = {k: np.ma.MaskedArray(data=x[k], mask=xm[k]) for k in x.keys()}
    return d


T = Union[np.ndarray, Dict[str, np.ndarray]]


def _concat_dict(xlist: List[Dict[str, T]]) -> Dict[str, T]:
    out_dict = {}
    for k, v in xlist[0].items():
        if isinstance(v, np.ndarray):
            out_dict[k] = np.concatenate([di[k] for di in xlist], axis=0)
        else:
            out_dict[k] = _concat_dict([di[k] for di in xlist])
    return out_dict


def _extract(xt: Dict[str, tf.Tensor], yt: tf.Tensor, sess: tf.Session) \
        -> Tuple[dict, np.ndarray]:

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
    x_full = _concat_dict(x_list)
    if "con" in x_full:
        x_full["con"] = _make_mask(x_full["con"], x_full["con_mask"])
    if "cat" in x_full:
        x_full["cat"] = _make_mask(x_full["cat"], x_full["cat_mask"])

    return x_full, y_full

def _get_data(records_train: List[str], records_test: List[str],
              metadata: Training, npoints: Optional[int],
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
              metadata: Training) \
        -> Iterator[Dict[str, np.ndarray]]:

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
                    n = X['indices'].shape[0]
                    pbar.update(n)
                    yield X
                except tf.errors.OutOfRangeError:
                    break
            return


def _split(x: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    x_con = x["con"] if "con" in x else None
    x_cat = x["cat"] if "cat" in x else None
    indices = x["indices"]
    coords = x["coords"]
    return x_con, x_cat, indices, coords

def train_test(config_module: str, records_train: List[str],
               records_test: List[str], metadata: Training,
               model_dir: str, maxpoints: Optional[int], batchsize: int,
               random_seed: int) -> None:

    log.info("Extracting and subsetting training data")
    data_tuple = _get_data(records_train, records_test, metadata, maxpoints,
                           batchsize, random_seed)
    x, y, x_test, y_test = data_tuple
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


def predict(modeldir: str, metadata: Training,
            query_records: List[str], batch_size: int) \
        -> Iterator[Dict[str, np.ndarray]]:

    model_path = os.path.join(modeldir, "skmodel.pickle")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    for xi in _query_it(query_records, batch_size, metadata):
        x_con, x_cat, indices, coords = _split(xi)
        res = model.predict(x_con, x_cat, indices, coords)
        yield res
