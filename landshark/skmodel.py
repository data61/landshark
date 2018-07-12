"""Scikit Learn training and testing with tf records."""
import logging
import os.path
import json

from typing import Optional, Tuple, List, Iterator
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, r2_score, \
    confusion_matrix

from landshark.metadata import TrainingMetadata, CategoricalMetadata
from landshark.model import train_data, test_data, sample_weights_labels
from landshark.serialise import deserialise
from landshark.basetypes import CategoricalType, OrdinalType, \
    RegressionPrediction, ClassificationPrediction, Prediction

log = logging.getLogger(__name__)


def _extract(Xo: tf.Tensor, Xom: tf.Tensor, Xc: tf.Tensor, Xcm: tf.Tensor,
             Y: tf.Tensor, sess: tf.Session, data_frac: Optional[float]=None,
             random_seed: int=666) \
        -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ndarray]:
    rnd = np.random.RandomState(random_seed)
    ord_list = []
    cat_list = []
    y_list = []

    has_ord = int(Xo.shape[1]) != 0
    has_cat = int(Xc.shape[1]) != 0

    try:
        while True:
            result = sess.run([Xo, Xom, Xc, Xcm, Y])
            x_ord_d, x_ord_m, x_cat_d, x_cat_m, y = result
            n = x_ord_d.shape[0] if has_ord else x_cat_d.shape[0]
            if data_frac is not None:
                mask = rnd.choice([True, False], size=(n,),
                                  p=[data_frac, 1.0 - data_frac])
            else:
                mask = slice(n)

            if has_ord:
                x_ord = np.ma.MaskedArray(data=x_ord_d, mask=x_ord_m)
                ord_list.append(x_ord[mask])
            if has_cat:
                x_cat = np.ma.MaskedArray(data=x_cat_d, mask=x_cat_m)
                cat_list.append(x_cat[mask])
            y_list.append(y[mask])
    except tf.errors.OutOfRangeError:
        pass
    ord_marray = None
    cat_marray = None
    if has_ord:
        ord_marray = np.ma.concatenate(ord_list, axis=0)
    if has_cat:
        cat_marray = np.ma.concatenate(cat_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)
    # sklearn only supports 1D Y at the moment
    assert y_array.ndim == 1 or y_array.shape[1] == 1
    y_array = y_array.flatten()
    return ord_marray, cat_marray, y_array


def _get_data(records_train: List[str], records_test: List[str],
              metadata: TrainingMetadata, npoints: Optional[int],
              batch_size: int,
              random_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
    data_frac = min(npoints / metadata.N, 1.0) if npoints else None

    train_dataset = train_data(records_train, batch_size, epochs=1,
                               random_seed=random_seed)
    test_dataset = test_data(records_test, batch_size)

    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes,
        shared_name="Iterator"
        )

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xo, Xom, Xc, Xcm, Y = deserialise(iterator, metadata)

    with tf.Session() as sess:
        sess.run(train_init_op)
        ord_array, cat_array, y_array = _extract(Xo, Xom, Xc, Xcm, Y, sess,
                                                 data_frac, random_seed)
        sess.run(test_init_op)
        ord_array_test, cat_array_test, y_array_test = _extract(Xo, Xom, Xc,
                                                                Xcm, Y, sess)

    return (ord_array, cat_array, y_array,
            ord_array_test, cat_array_test, y_array_test)


def _query_it(records_query: List[str], batch_size: int,
              metadata: TrainingMetadata) \
        -> Iterator[Tuple[np.ma.MaskedArray, np.ma.MaskedArray]]:

    total_size = metadata.features.image.height * metadata.features.image.width
    dataset = test_data(records_query, batch_size)
    iterator = dataset.make_one_shot_iterator()
    Xo, Xom, Xc, Xcm, Y = deserialise(iterator, metadata)

    has_ord = int(Xo.shape[1]) != 0
    has_cat = int(Xc.shape[1]) != 0

    with tqdm(total=total_size) as pbar:
        with tf.Session() as sess:
            while True:
                try:
                    xo, xom, xc, xcm = sess.run([Xo, Xom, Xc, Xcm])
                    ord_marray = np.ma.MaskedArray(data=xo, mask=xom) \
                        if has_ord else None
                    cat_marray = np.ma.MaskedArray(data=xc, mask=xcm) \
                        if has_cat else None
                    n = xo.shape[0] if has_ord else xc.shape[0]
                    pbar.update(n)
                    yield ord_marray, cat_marray
                except tf.errors.OutOfRangeError:
                    break
            return


def _convert_res(res: Tuple[np.ndarray, Optional[np.ndarray]]) -> Prediction:
    """Make sure Y adheres to our conventions."""
    # regression
    y, extra = res
    if extra is not None:
        extra = extra.astype(OrdinalType)
    if y.dtype == np.float64 or y.dtype == np.float32:
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = y.astype(OrdinalType)
        if extra is not None:
            if extra.shape[0] != 2:
                raise RuntimeError("The regressor must output either None or "
                                   "upper and lower quantiles in 2xN array.")

            # Add another dimension (percentiles are expected in batches)
            if extra.ndim == 2:
                extra = extra[..., None]

        out: Prediction = RegressionPrediction(Ey=y, percentiles=extra)

    elif y.dtype == np.int64 or y.dtype == np.int32:
        y = y.astype(CategoricalType)
        # Make binary classifier output consistent with TensorFlow
        if extra is not None and np.ndim(extra) < 2:
            raise RuntimeError("The classifier needs to output E[y] and p(y)!")
        out = ClassificationPrediction(Ey=y, probabilities=extra)
    return out


def train_test(config_module: str, records_train: List[str],
               records_test: List[str], metadata: TrainingMetadata,
               model_dir: str, maxpoints: Optional[int], batchsize: int,
               random_seed: int) -> None:

    classification = isinstance(metadata.targets, CategoricalMetadata)

    log.info("Extracting and subsetting training data")
    data_tuple = _get_data(records_train, records_test, metadata, maxpoints,
                           batchsize, random_seed)
    ord_array, cat_array, y_array, \
        ord_array_test, cat_array_test, y_array_test = data_tuple

    userconfig = __import__(config_module)

    log.info("Training model")
    model = userconfig.SKModel(metadata, random_seed=random_seed)

    model.fit(ord_array, cat_array, y_array)
    log.info("Evaluating test data")
    res = model.predict(ord_array_test, cat_array_test, None)
    res = _convert_res(res)

    if classification:
        EYs, pys = res
        sample_weights, labels = sample_weights_labels(metadata, EYs)
        acc = accuracy_score(y_array_test, EYs)
        bacc = accuracy_score(y_array_test, EYs, sample_weight=sample_weights)
        conf = confusion_matrix(y_array_test, EYs)
        nlabels = metadata.targets.ncategories
        labels = np.arange(nlabels)
        lp = -1 * log_loss(y_array_test, pys, labels=labels)
        log.info("Sklearn acc: {:.5f}, lp: {:.5f}".format(acc, lp))
        scores = {"acc": acc, "bacc": bacc, "lp": lp, "confmat": conf.tolist()}

    else:
        EYs, _ = res
        r2_arr = r2_score(y_array_test, EYs, multioutput="raw_values")
        if r2_arr.size == 1:
            r2 = r2_arr[0]
        else:
            r2 = list(r2_arr)

        log.info("Sklearn r2: {}" .format(r2))
        scores = {"r2": r2.tolist()}

    log.info("Saving model to disk")
    model_path = os.path.join(model_dir, "skmodel.pickle")
    with open(model_path, "wb") as fb:
        pickle.dump(model, fb)

    score_path = os.path.join(model_dir, "skmodel.json")
    with open(score_path, "w") as f:
        json.dump(scores, f)


def predict(modeldir: str, metadata: TrainingMetadata,
            query_records: List[str], batch_size: int,
            percentiles: Tuple[float, float]) -> Iterator[Prediction]:

    model_path = os.path.join(modeldir, "skmodel.pickle")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    for xo, xc in _query_it(query_records, batch_size, metadata):
        res = model.predict(xo, xc, percentiles)
        res = _convert_res(res)
        yield res
