import logging

import tensorflow as tf
import numpy as np

from landshark import model
from landshark.feed import query_data
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score

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
    test_dataset = model.test_data(records_test, batch_size, 1)

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


def train_test_predict(records_train, records_test, records_query,
                       metadata, npoints, trees, batch_size=10000,
                       random_seed=666):

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                  verbose=0, copy=True)
    enc = OneHotEncoder(n_values=np.array(metadata.ncategories),
                        categorical_features="all", dtype=np.float32,
                        sparse=False)
    est = RandomForestClassifier(n_estimators=trees) if \
        metadata.target_dtype == np.int32 else \
        RandomForestRegressor(n_estimators=trees)

    data_tuple = _get_data(records_train, records_test, metadata, npoints,
                           batch_size, random_seed)

    ord_array, cat_array, y_array, \
        ord_array_test, cat_array_test, y_array_test = data_tuple

    X_onehot = enc.fit_transform(cat_array)
    X_imputed = imp.fit_transform(ord_array)
    X = np.concatenate([X_onehot, X_imputed], axis=1)
    log.info("Training random forest")
    est.fit(X, y_array)

    def predict(xo, xc):
        x_onehot = enc.transform(xc)
        x_imputed = imp.transform(xo)
        x = np.concatenate([x_onehot, x_imputed], axis=1)
        y_star = est.predict(x)
        return y_star.astype(np.float32)

    y_star = predict(ord_array_test, cat_array_test)
    scores = r2_score(y_array_test, y_star, multioutput='raw_values')
    log.info("Random forest R2: {}".format(scores))

    # predict over image
    qdata = query_data(features, batch_size, metadata.halfwidth)
    for d in qdata:
        x_ord = d.x_ord.reshape((d.x_ord.shape[0], -1))
        x_cat = d.x_cat.reshape((d.x_cat.shape[0], -1))
        x_ord.data[x_ord.mask] = np.nan
        ys = predict(x_ord.data, x_cat.data)
        std = np.zeros_like(ys, dtype=np.float32)
        yield ys, std
