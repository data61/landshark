import logging

import tensorflow as tf
import numpy as np

from landshark import model
from landshark.feed import query_data
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)


def _get_data(records_train, records_test, metadata, npoints,
              batch_size, random_seed):
    data_frac = npoints / metadata.N
    rnd = np.random.RandomState(random_seed)

    train_dataset = model.train_data(records_train, batch_size, epochs=1)
    test_dataset = model.test_data(records_test, batch_size, 1)

    with tf.name_scope("Sources"):
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes,
            shared_name="Iterator"
            )
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xo, Xom, Xc, Xcm, Y = model.decode(iterator, metadata)

    ord_list = []
    cat_list = []
    y_list = []

    with tf.Session() as sess:
        sess.run(train_init_op)
        try:
            while True:
                result = sess.run([Xo, Xom, Xc, Xcm, Y])
                x_ord, x_ord_mask, x_cat, x_cat_mask, y = result
                xm_ord = np.ma.MaskedArray(data=x_ord, mask=x_ord_mask)
                xm_cat = np.ma.MaskedArray(data=x_cat, mask=x_cat_mask)
                n = x_ord.shape[0]
                mask = rnd.choice([True, False], size=(n,),
                                  p=[data_frac, 1.0 - data_frac])
                ord_list.append(xm_ord[mask])
                cat_list.append(xm_cat[mask])
                y_list.append(y[mask])
        except tf.errors.OutOfRangeError:
            log.info("Training data extraction complete")
            pass
        ord_array = np.ma.concatenate(ord_list, axis=0)
        cat_array = np.ma.concatenate(cat_list, axis=0)
        y_array = np.concatenate(y_list, axis=0)
        ord_list = []
        cat_list = []
        y_list = []

        sess.run(test_init_op)
        try:
            while True:
                result = sess.run([Xo, Xom, Xc, Xcm, Y])
                x_ord, x_ord_mask, x_cat, x_cat_mask, y = result
                xm_ord = np.ma.MaskedArray(data=x_ord, mask=x_ord_mask)
                xm_cat = np.ma.MaskedArray(data=x_cat, mask=x_cat_mask)
                ord_list.append(xm_ord)
                cat_list.append(xm_cat)
                y_list.append(y)
        except tf.errors.OutOfRangeError:
            log.info("Training data extraction complete")
            pass
        ord_array_test = np.ma.concatenate(ord_list, axis=0)
        cat_array_test = np.ma.concatenate(cat_list, axis=0)
        y_array_test = np.concatenate(y_list, axis=0)


    return (ord_array, cat_array, y_array,
            ord_array_test, cat_array_test, y_array_test)


def train_test_predict(records_train, records_test, metadata,
                       features, npoints, trees, batch_size=10000,
                       random_seed=666):

    res = _get_data(records_train, records_test, metadata, npoints,
                    batch_size, random_seed)

    ord_array, cat_array, y_array, \
        ord_array_test, cat_array_test, y_array_test = res
    ord_array.data[ord_array.mask] = np.nan

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                  verbose=0, copy=True)
    enc = OneHotEncoder(n_values=np.array(metadata.ncategories),
                        categorical_features="all", dtype=np.float32,
                        sparse=False)
    X_onehot = enc.fit_transform(cat_array.data)
    X_imputed = imp.fit_transform(ord_array.data)
    X = np.concatenate([X_onehot, X_imputed], axis=1)

    if y_array.dtype == np.int32:
        est = RandomForestClassifier(n_estimators=trees)
    else:
        est = RandomForestRegressor(n_estimators=trees)
    log.info("Training random forest")
    est.fit(X, y_array)

    Xs_onehot = enc.transform(cat_array_test.data)
    Xs_imputed = imp.transform(ord_array_test.data)
    Xs = np.concatenate([Xs_onehot, Xs_imputed], axis=1)
    log.info("Evaluating RF on test data")
    y_star = est.predict(Xs)

    scores = r2_score(y_array_test, y_star, multioutput='raw_values')
    log.info("Random forest R2: {}".format(scores))

    # predictions now??
    qdata = query_data(features, batch_size, metadata.halfwidth)
    for d in qdata:
        x_ord = d.x_ord.reshape((d.x_ord.shape[0], -1))
        x_cat = d.x_cat.reshape((d.x_cat.shape[0], -1))
        x_ord.data[x_ord.mask] = np.nan
        X_onehot = enc.transform(x_cat.data)
        X_imputed = imp.transform(x_ord.data)
        X = np.concatenate([X_onehot, X_imputed], axis=1)
        ys = est.predict(X).astype(np.float32)
        std = np.zeros_like(ys, dtype=np.float32)
        yield ys, std
