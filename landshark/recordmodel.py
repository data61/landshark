"""Train/test with tfrecords."""

from typing import List
import tensorflow as tf


fdict = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_cat_shape": tf.FixedLenFeature(3, tf.int64),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "x_ord_shape": tf.FixedLenFeature(3, tf.int64),
    "y": tf.FixedLenFeature([], tf.string)
    }

def train_test(training_records: List[str],
               testing_records: List[str]) -> None:
    """Train and test."""
    dataset = tf.data.TFRecordDataset(training_records)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    raw_feature = tf.parse_single_example(next_element, features=fdict)

    x_ord_flat = tf.decode_raw(raw_feature["x_ord"], tf.float32)
    x_cat_flat = tf.decode_raw(raw_feature["x_cat"], tf.int32)
    x_ord_shape = raw_feature["x_ord_shape"]
    x_cat_shape = raw_feature["x_cat_shape"]
    x_ord = tf.reshape(x_ord_flat, x_ord_shape)
    x_cat = tf.reshape(x_cat_flat, x_cat_shape)

    with tf.Session() as sess:
        ord_val = sess.run(x_ord)
        cat_val = sess.run(x_cat)
        import IPython; IPython.embed(); import sys; sys.exit()
