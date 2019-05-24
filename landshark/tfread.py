"""Import data from tensorflow format."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from glob import glob
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from landshark.metadata import FeatureSet, Target, Training
from landshark.serialise import deserialise
from landshark.util import mb_to_points

log = logging.getLogger(__name__)


def dataset_fn(
    records: List[str],
    batchsize: int,
    features: FeatureSet,
    targets: Optional[Target] = None,
    epochs: int = 1,
    take: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None
) -> Callable[[], tf.data.TFRecordDataset]:
    """Dataset feeder."""
    def f() -> tf.data.TFRecordDataset:
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB") \
            .repeat(count=epochs)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer, seed=random_seed
            )
        dataset = dataset.take(take) \
            .batch(batchsize) \
            .map(lambda x: deserialise(x, features, targets))
        return dataset
    return f


def get_training_meta(directory: str) -> Tuple[Training, List[str], List[str]]:
    """Read train/test metadata and record filenames from dir."""
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))
    metadata = Training.load(directory)
    return metadata, training_records, testing_records


def get_query_meta(query_dir: str) -> Tuple[FeatureSet, List[str], int, int]:
    """Read query metadata and record filenames from dir."""
    strip_list = query_dir.split("strip")[-1].split("of")
    assert len(strip_list) == 2
    strip = int(strip_list[0])
    nstrip = int(strip_list[1])
    query_metadata = FeatureSet.load(query_dir)
    query_records = glob(os.path.join(query_dir, "*.tfrecord"))
    query_records.sort()
    return query_metadata, query_records, strip, nstrip


def _make_mask(x: Dict[str, np.ndarray],
               xm: Dict[str, np.ndarray]
               ) -> Dict[str, np.ma.MaskedArray]:
    assert x.keys() == xm.keys()
    d = {k: np.ma.MaskedArray(data=x[k], mask=xm[k]) for k in x.keys()}
    return d


T = Union[np.ndarray, Dict[str, np.ndarray]]


def _concat_dict(xlist: List[Dict[str, T]]) -> Dict[str, T]:
    out_dict = {}
    for k, v in xlist[0].items():
        if isinstance(v, np.ndarray):
            out_dict[k] = np.concatenate([di[k] for di in xlist], axis=0)
        else:
            out_dict[k] = _concat_dict([di[k] for di in xlist])
    return out_dict


def _extract(xt: Dict[str, tf.Tensor],
             yt: tf.Tensor,
             sess: tf.Session
             ) -> Tuple[dict, np.ndarray]:

    x_list = []
    y_list = []
    try:
        while True:
            x, y = sess.run([xt, yt])
            x_list.append(x)
            y_list.append(y)
    except tf.errors.OutOfRangeError:
        pass

    y_full = np.concatenate(y_list, axis=0)
    x_full = _concat_dict(x_list)
    if "con" in x_full:
        x_full["con"] = _make_mask(x_full["con"], x_full["con_mask"])
        x_full.pop("con_mask")
    if "cat" in x_full:
        x_full["cat"] = _make_mask(x_full["cat"], x_full["cat_mask"])
        x_full.pop("cat_mask")

    return x_full, y_full


def _split(x: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    x_con = x["con"] if "con" in x else None
    x_cat = x["cat"] if "cat" in x else None
    indices = x["indices"]
    coords = x["coords"]
    return x_con, x_cat, indices, coords


def extract_split_xy(
    dataset: tf.data.TFRecordDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, Y) data from tensor dataset and split."""
    X_tensor, Y_tensor = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        X, Y = _extract(X_tensor, Y_tensor, sess)
    x_con, x_cat, indices, coords = _split(X)
    return x_con, x_cat, indices, coords, Y


def xy_record_data(
    records: List[str],
    metadata: Training,
    batchsize: int = 1000,
    npoints: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Read train/test record."""
    train_dataset = dataset_fn(
        records=records,
        batchsize=batchsize,
        features=metadata.features,
        targets=metadata.targets,
        epochs=1,
        take=npoints,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        random_seed=random_seed
    )()
    xy_data_tuple = extract_split_xy(train_dataset)
    return xy_data_tuple


# TODO simplify now I'm no longer using the recursive dict structure

def query_data_it(
    records_query: List[str],
    batch_size: int,
    features: FeatureSet
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    dataset = dataset_fn(records_query, batch_size, features)()
    X_tensor = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        while True:
            try:
                X = sess.run(X_tensor)
                if "con" in X:
                    X["con"] = _make_mask(X["con"], X["con_mask"])
                if "cat" in X:
                    X["cat"] = _make_mask(X["cat"], X["cat_mask"])
                yield _split(X)
            except tf.errors.OutOfRangeError:
                break
        return


def read_train_record(
    directory: str,
    npoints: int = -1,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Read train record."""
    metadata, train_records, _ = get_training_meta(directory)
    train_data_tuple = xy_record_data(
        records=train_records,
        metadata=metadata,
        npoints=npoints,
        shuffle=(random_seed is not None),
        random_seed=random_seed
    )
    return train_data_tuple


def read_test_record(
    directory: str,
    npoints: int = -1,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Read test record."""
    metadata, _, test_records = get_training_meta(directory)
    test_data_tuple = xy_record_data(
        records=test_records,
        metadata=metadata,
        npoints=npoints,
        shuffle=(random_seed is not None),
        random_seed=random_seed
    )
    return test_data_tuple


def read_query_record(
    query_dir: str, batch_mb: float,
) -> Iterator[np.ndarray]:
    features, query_records, strip, nstrip = get_query_meta(query_dir)
    ndims_con = len(features.continuous) if features.continuous else 0
    ndims_cat = len(features.categorical) if features.categorical else 0
    points_per_batch = mb_to_points(
        batch_mb, ndims_con, ndims_cat, features.halfwidth
    )
    yield from query_data_it(query_records, points_per_batch, features)
