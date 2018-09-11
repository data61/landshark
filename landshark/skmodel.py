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


def _extract(Xo: tf.Tensor, Xom: tf.Tensor, Xc: tf.Tensor, Xcm: tf.Tensor,
             Y: tf.Tensor, sess: tf.Session, data_frac: Optional[float]=None,
             random_seed: int=666) \
        -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ndarray]:
    rnd = np.random.RandomState(random_seed)
    con_list = []
    cat_list = []
    y_list = []

    has_con = int(Xo.shape[1]) != 0
    has_cat = int(Xc.shape[1]) != 0

    try:
        while True:
            result = sess.run([Xo, Xom, Xc, Xcm, Y])
            x_con_d, x_con_m, x_cat_d, x_cat_m, y = result
            n = x_con_d.shape[0] if has_con else x_cat_d.shape[0]
            if data_frac is not None:
                mask = rnd.choice([True, False], size=(n,),
                                  p=[data_frac, 1.0 - data_frac])
            else:
                mask = slice(n)

            if has_con:
                x_con = np.ma.MaskedArray(data=x_con_d, mask=x_con_m)
                con_list.append(x_con[mask])
            if has_cat:
                x_cat = np.ma.MaskedArray(data=x_cat_d, mask=x_cat_m)
                cat_list.append(x_cat[mask])
            y_list.append(y[mask])
    except tf.errors.OutOfRangeError:
        pass
    con_marray = None
    cat_marray = None
    if has_con:
        con_marray = np.ma.concatenate(con_list, axis=0)
    if has_cat:
        cat_marray = np.ma.concatenate(cat_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)
    y_array = np.squeeze(y_array)
    return con_marray, cat_marray, y_array


def _get_data(records_train: List[str], records_test: List[str],
              metadata: TrainingMetadata, npoints: Optional[int],
              batch_size: int,
              random_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
    data_frac = min(npoints / metadata.N, 1.0) if npoints else None


    X, Y = train_data(records_train, metadata, batch_size,
                      epochs=1, random_seed=random_seed)()
    Xt, Yt = test_data(records_test, metadata, batch_size)()

    with tf.Session() as sess:
        con_array, cat_array, y_array = _extract(X['con'], X['con_mask'],
                                                 X['cat'], X['cat_mask'],
                                                 Y, sess, data_frac,
                                                 random_seed)
        con_array_test, cat_array_test, y_array_test = _extract(Xt['con'],
                                                                Xt['con_mask'],
                                                                Xt['cat'],
                                                                Xt['cat_mask'],
                                                                Yt, sess)
    return (con_array, cat_array, y_array,
            con_array_test, cat_array_test, y_array_test)


def _query_it(records_query: List[str], batch_size: int,
              metadata: TrainingMetadata) \
        -> Iterator[Tuple[np.ma.MaskedArray, np.ma.MaskedArray]]:

    total_size = metadata.features.image.height * metadata.features.image.width
    X = predict_data(records_query, metadata, batch_size)()
    has_con = int(X['con'].shape[1]) != 0
    has_cat = int(X['cat'].shape[1]) != 0

    with tqdm(total=total_size) as pbar:
        with tf.Session() as sess:
            while True:
                try:
                    Xvals = sess.run(X)
                    con_marray = np.ma.MaskedArray(data=Xvals['con'],
                                                   mask=Xvals['con_mask']) \
                        if has_con else None
                    cat_marray = np.ma.MaskedArray(data=Xvals['cat'],
                                                   mask=Xvals['cat_mask']) \
                        if has_cat else None
                    n = con_marray.shape[0] if has_con else cat_marray.shape[0]
                    pbar.update(n)
                    yield con_marray, cat_marray
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
    con_array, cat_array, y_array, \
        con_array_test, cat_array_test, y_array_test = data_tuple

    userconfig = __import__(config_module)

    log.info("Training model")
    model = userconfig.SKModel(metadata, random_seed=random_seed)

    model.train(con_array, cat_array, y_array)
    log.info("Evaluating test data")
    res = model.predict(con_array_test, cat_array_test)
    scores = model.test(y_array_test, res)
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

    for xo, xc in _query_it(query_records, batch_size, metadata):
        res = model.predict(xo, xc)
        yield res
