"""Scikit Learn training and testing with tf records."""
import logging
import os.path

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pickle

from landshark.model import train_data, test_data
from landshark.serialise import deserialise
from landshark.basetypes import CategoricalType, OrdinalType

from sklearn.metrics import accuracy_score, log_loss, r2_score


log = logging.getLogger(__name__)


def _extract(Xo, Xom, Xc, Xcm, Y, sess, data_frac=None, random_seed=666):
    rnd = np.random.RandomState(random_seed)
    ord_list = []
    cat_list = []
    y_list = []

    has_ord = int(Xo.shape[1]) != 0
    has_cat = int(Xc.shape[1]) != 0

    try:
        while True:
            result = sess.run([Xo, Xom, Xc, Xcm, Y])
            x_ord_data, x_ord_mask, x_cat, _, y = result
            n = x_ord_data.shape[0]
            if has_ord:
                x_ord = np.ma.MaskedArray(data=x_ord_data, mask=x_ord_mask)
            if data_frac is not None:
                mask = rnd.choice([True, False], size=(n,),
                                  p=[data_frac, 1.0 - data_frac])
            else:
                mask = slice(n)

            if has_ord:
                ord_list.append(x_ord[mask])
            if has_cat:
                cat_list.append(x_cat[mask])
            y_list.append(y[mask])
    except tf.errors.OutOfRangeError:
        pass
    ord_array = None
    cat_array = None
    if has_ord:
        ord_array = np.ma.concatenate(ord_list, axis=0)
        ord_array.data[ord_array.mask] = np.nan
        ord_array = ord_array.data
    if has_cat:
        cat_array = np.concatenate(cat_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)
    return ord_array, cat_array, y_array


def _get_data(records_train, records_test, metadata, npoints,
              batch_size, random_seed):
    data_frac = min(npoints / metadata.N, 1.0)

    train_dataset = train_data(records_train, batch_size, epochs=1)
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


def _query_it(records_query, batch_size, metadata):

    total_size = metadata.image_spec.height * metadata.image_spec.width
    dataset = test_data(records_query, batch_size)
    iterator = dataset.make_one_shot_iterator()
    Xo, Xom, Xc, Xcm, Y = deserialise(iterator, metadata)

    with tqdm(total=total_size) as pbar:
        with tf.Session() as sess:
            while True:
                try:
                    xo, xom, xc, xcm = sess.run([Xo, Xom, Xc, Xcm])
                    ord_array = xo
                    ord_array[xom] = np.nan
                    cat_array = xc
                    pbar.update(xo.shape[0])
                    yield ord_array, cat_array
                except tf.errors.OutOfRangeError:
                    break
            return


def _convert_res(res):
    """Make sure Y adheres to our conventions."""
    y, extra = res
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.dtype == np.float64 or y.dtype == np.float32:
        y = y.astype(OrdinalType)
    elif y.dtype == np.int64 or y.dtype == np.int32:
        y = y.astype(CategoricalType)
    return y, extra


def train_test(config_module, records_train, records_test, metadata, model_dir,
               maxpoints, batchsize, random_seed):

    classification = metadata.target_dtype != OrdinalType

    log.info("Extracting and subsetting training data")
    data_tuple = _get_data(records_train, records_test, metadata, maxpoints,
                           batchsize, random_seed)
    ord_array, cat_array, y_array, \
        ord_array_test, cat_array_test, y_array_test = data_tuple

    userconfig = __import__(config_module)

    log.info("Training model")
    model = userconfig.SKModel(metadata)
    model.fit(ord_array, cat_array, y_array)
    log.info("Evaluating test data")
    res = model.predict(ord_array_test, cat_array_test)
    res = _convert_res(res)

    if classification:
        EYs, pys = res
        acc = accuracy_score(y_array_test, EYs)
        nlabels = len(metadata.target_map[0])
        labels = np.arange(nlabels)
        lp = -1 * log_loss(y_array_test, pys, labels=labels)
        log.info("Sklearn acc: {:.5f}, lp: {:.5f}".format(acc, lp))

    else:
        EYs, _ = res
        r2 = r2_score(y_array_test, EYs, multioutput='raw_values')
        log.info("Sklearn r2: {}" .format(r2))

    log.info("Saving model to disk")
    model_path = os.path.join(model_dir, "skmodel.pickle")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def predict(modeldir, metadata, query_records, batch_size):

    model_path = os.path.join(modeldir, "skmodel.pickle")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    for xo, xc in _query_it(query_records, batch_size, metadata):
        res = model.predict(xo, xc)
        res = _convert_res(res)
        yield res
