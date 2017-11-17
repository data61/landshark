import logging
import os.path

import tensorflow as tf
import numpy as np
import pickle

from landshark import model
from landshark.feed import query_data
from sklearn.metrics import accuracy_score, log_loss, r2_score
import scipy.stats

log = logging.getLogger(__name__)


def _extract(Xo, Xom, Xc, Xcm, Y, sess, data_frac=None, random_seed=666):
    rnd = np.random.RandomState(random_seed)
    ord_list = []
    cat_list = []
    y_list = []
    try:
        while True:
            result = sess.run([Xo, Xom, Xc, Xcm, Y])
            x_ord_data, x_ord_mask, x_cat, _, y = result
            x_ord = np.ma.MaskedArray(data=x_ord_data, mask=x_ord_mask)
            n = x_ord.shape[0]
            if data_frac is not None:
                mask = rnd.choice([True, False], size=(n,),
                                  p=[data_frac, 1.0 - data_frac])
                ord_list.append(x_ord[mask])
                cat_list.append(x_cat[mask])
                y_list.append(y[mask])
            else:
                ord_list.append(x_ord)
                cat_list.append(x_cat)
                y_list.append(y)
    except tf.errors.OutOfRangeError:
        pass
    ord_array = np.ma.concatenate(ord_list, axis=0)
    ord_array.data[ord_array.mask] = np.nan
    ord_array = ord_array.data
    cat_array = np.concatenate(cat_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)
    return ord_array, cat_array, y_array


def _get_data(records_train, records_test, metadata, npoints,
              batch_size, random_seed):
    data_frac = min(npoints / metadata.N, 1.0)

    train_dataset = model.train_data(records_train, batch_size, epochs=1)
    test_dataset = model.test_data(records_test, batch_size)

    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes,
        shared_name="Iterator"
    )

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xo, Xom, Xc, Xcm, Y = model.decode(iterator, metadata)

    with tf.Session() as sess:
        sess.run(train_init_op)
        ord_array, cat_array, y_array = _extract(Xo, Xom, Xc, Xcm, Y,
            sess, data_frac, random_seed)
        sess.run(test_init_op)
        ord_array_test, cat_array_test, y_array_test = _extract(Xo, Xom, Xc,
            Xcm, Y, sess)

    return (ord_array, cat_array, y_array,
            ord_array_test, cat_array_test, y_array_test)


def train_test(config_module, records_train, records_test, metadata, model_dir,
               maxpoints, batchsize, random_seed):

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
    EYs = model.predict(ord_array_test, cat_array_test)

    r2 = r2_score(y_array_test, EYs, multioutput='raw_values')
    log.info("Sklearn r2: {}" .format(r2))

    log.info("Saving model to disk")
    model_path = os.path.join(model_dir, "skmodel.pickle")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def predict():
    pass

#     def predict(xo, xc):
#         x_onehot = enc.transform(xc)
#         x_imputed = imp.transform(xo)
#         x = np.concatenate([x_onehot, x_imputed], axis=1)
#         y_star = est.predict(x)
#         if np.ndim(y_star) is 1:
#             y_star = y_star[:, np.newaxis]
#         return y_star.astype(np.float32)

#     y_star = predict(ord_array_test, cat_array_test)
#     scores = r2_score(y_array_test, y_star, multioutput='raw_values')
#     log.info("Random forest R2: {}".format(scores))

#     # predict over image
#     qdata = query_data(features, batch_size, metadata.halfwidth)
#     for d in qdata:
#         x_ord = d.x_ord.reshape((d.x_ord.shape[0], -1))
#         x_cat = d.x_cat.reshape((d.x_cat.shape[0], -1))
#         x_ord.data[x_ord.mask] = np.nan
#         ys = predict(x_ord.data, x_cat.data)
#         ysl = np.zeros_like(ys, dtype=np.float32)
#         ysu = np.zeros_like(ys, dtype=np.float32)
#         yield ys, ysl, ysu
