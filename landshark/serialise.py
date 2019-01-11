"""Serialise and Deserialise to and from tf records."""

from itertools import repeat
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from landshark.basetypes import CategoricalType
from landshark.metadata import Feature, Training

#
# Module constants and types
#

_FDICT = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_con": tf.FixedLenFeature([], tf.string),
    "x_con_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string),
    "indices": tf.FixedLenFeature([], tf.string),
    "coords": tf.FixedLenFeature([], tf.string)
    }


class DataArrays(NamedTuple):
    con_marray: Optional[np.ma.MaskedArray]
    cat_marray: Optional[np.ma.MaskedArray]
    targets: Optional[np.ndarray]
    world_coords: np.ndarray
    image_indices: np.ndarray

#
# Module functions
#


def serialise(x: DataArrays) -> List[bytes]:
    """Serialise data to tf.records."""
    x_con = repeat(np.ma.MaskedArray(data=[], mask=[])) \
        if x.con_marray is None else x.con_marray
    x_cat = repeat(np.ma.MaskedArray(data=[], mask=[])) \
        if x.cat_marray is None else x.cat_marray
    y = repeat(np.array([])) if x.targets is None else x.targets
    indices = x.image_indices
    coords = x.world_coords

    string_list = []
    for xo_i, xc_i, y_i, idx_i, c_i in zip(x_con, x_cat, y, indices, coords):
        fdict = _make_features(xo_i, xc_i, y_i, idx_i, c_i)
        example = tf.train.Example(
            features=tf.train.Features(feature=fdict))
        string_list.append(example.SerializeToString())
    return string_list


def deserialise(row: str,
                metadata: Training,
                ignore_y: bool = False
                ) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Decode tf.record strings into Tensors."""
    raw_features = tf.parse_example(row, features=_FDICT)
    npatch_side = 2 * metadata.features.halfwidth + 1
    categorical = metadata.targets.dtype == CategoricalType
    y_type = tf.int32 if categorical else tf.float32
    with tf.name_scope("Inputs"):
        x_con = tf.decode_raw(raw_features["x_con"], tf.float32)
        x_cat = tf.decode_raw(raw_features["x_cat"], tf.int32)
        x_con_mask = tf.decode_raw(raw_features["x_con_mask"], tf.uint8)
        x_cat_mask = tf.decode_raw(raw_features["x_cat_mask"], tf.uint8)
        x_con_mask = tf.cast(x_con_mask, tf.bool)
        x_cat_mask = tf.cast(x_cat_mask, tf.bool)
        y = tf.decode_raw(raw_features["y"], y_type)
        indices = tf.decode_raw(raw_features["indices"], tf.int32)
        coords = tf.decode_raw(raw_features["coords"], tf.float64)
        ntargets = metadata.targets.D

        y.set_shape((None, ntargets))
        indices.set_shape((None, 2))
        coords.set_shape((None, 2))

        feat_dict = {"indices": indices,
                     "coords": coords}

        if metadata.features.continuous:
            feat_dict["con"] = _unpack(x_con,
                                       metadata.features.continuous.columns,
                                       npatch_side)
            feat_dict["con_mask"] = _unpack(
                x_con_mask,
                metadata.features.continuous.columns,
                npatch_side)
        if metadata.features.categorical:
            feat_dict["cat"] = _unpack(x_cat,
                                       metadata.features.categorical.columns,
                                       npatch_side)
            feat_dict["cat_mask"] = _unpack(
                x_cat_mask,
                metadata.features.categorical.columns,
                npatch_side)

    result = feat_dict if ignore_y else (feat_dict, y)
    return result


def _unpack(x: tf.Tensor, columns: Dict[str, Feature], npatch_side: int) \
        -> Dict[str, tf.Tensor]:
    nfeatures = len(columns)
    x_all = tf.reshape(x, (tf.shape(x)[0], npatch_side,
                           npatch_side, nfeatures))
    start = 0
    stop = 0
    d = {}
    for k, v in columns.items():
        stop = start + v.D
        d[k] = x_all[..., start:stop]
        start = stop
    return d

#
# Private module utilities
#


def _ndarray_feature(x: np.ndarray) -> tf.train.Feature:
    """Create an ndarray feature stored as bytes."""
    x_bytes = x.tostring()
    feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[x_bytes]))
    return feature


def _make_features(x_con: np.ma.MaskedArray,
                   x_cat: np.ma.MaskedArray,
                   y: np.ndarray,
                   idx: np.ndarray,
                   coords: np.ndarray
                   ) -> dict:
    """Do stuff."""
    fdict = {
        "x_cat": _ndarray_feature(x_cat.data),
        "x_cat_mask": _ndarray_feature(x_cat.mask),
        "x_con": _ndarray_feature(x_con.data),
        "x_con_mask": _ndarray_feature(x_con.mask),
        "y": _ndarray_feature(y),
        "indices": _ndarray_feature(idx),
        "coords": _ndarray_feature(coords)
        }
    return fdict
