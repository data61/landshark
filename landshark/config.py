import tensorflow as tf
import numpy as np

def flatten_patch(x):
    new_shp = (tf.shape(x)[0], np.product(x.shape[1:]))
    new_x = tf.reshape(x, new_shp)
    return new_x

def value_impute(data, mask, newval):
    tmask = tf.cast(mask, dtype=data.dtype)
    fmask = tf.cast(tf.logical_not(mask), dtype=data.dtype)
    newdata = data * fmask + newval * tmask
    return newdata

def continuous_input(d):
    cols = [tf.feature_column.numeric_column(k)
                   for k in d.keys()]
    inputs = tf.feature_column.input_layer(d, cols)
    return inputs

def categorical_embedded_input(d, ncat_dict, embed_dict):
    columns_cat = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
        key=k, num_buckets=(v + 1)), embed_dict[k])
        for k, v in ncat_dict.items()]
    inputs_cat = tf.feature_column.input_layer(d, columns_cat)
    return inputs_cat


