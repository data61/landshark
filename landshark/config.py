import tensorflow as tf
import numpy as np

def flatten_patch(x):
    new_shp = (tf.shape(x)[0], np.product(x.shape[1:]))
    new_x = tf.reshape(x, new_shp)
    return new_x

