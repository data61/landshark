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
from importlib.util import module_from_spec, spec_from_file_location
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from landshark.model import test_data, train_data
from landshark.metadata import FeatureSet, Training

log = logging.getLogger(__name__)


def _load_config(module_name: str, path: str) -> None:
    # Load the model
    modspec = spec_from_file_location(module_name, path)
    cf = module_from_spec(modspec)
    if not modspec.loader:
        raise RuntimeError("Could not load configuration module")
    modspec.loader.exec_module(cf)  # type: ignore
    # needed for pickling??
    sys.modules[module_name] = cf


def load_model(config_file: str) -> str:
    module_name = "userconfig"
    _load_config(module_name, config_file)
    return module_name


def get_training_meta(directory: str) -> Tuple[List[str], List[str], Training]:
    """Read training metadata and record filenames from dir."""
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))
    metadata = Training.load(directory)
    return training_records, testing_records, metadata


def setup_training(config: str,
                   directory: str
                   ) -> Tuple[List[str], List[str], Training, str, str]:
    # Get the metadata
    training_records, testing_records, metadata = get_training_meta(directory)

    # Write the metadata
    name = os.path.basename(config).rsplit(".")[0] + \
        "_model_{}of{}".format(metadata.testfold, metadata.nfolds)
    model_dir = os.path.join(os.getcwd(), name)
    try:
        os.makedirs(model_dir)
    except FileExistsError:
        pass
    metadata.save(model_dir)

    # Load the model
    module_name = load_model(config)

    return training_records, testing_records, metadata, model_dir, module_name


def setup_query(config: str,
                querydir: str,
                checkpoint: str
                ) -> Tuple[Training, FeatureSet, List[str], int, int, str]:
    strip_list = querydir.split("strip")[-1].split("of")
    assert len(strip_list) == 2
    strip = int(strip_list[0])
    nstrip = int(strip_list[1])

    query_metadata = FeatureSet.load(querydir)
    training_metadata = Training.load(checkpoint)
    query_records = glob(os.path.join(querydir, "*.tfrecord"))
    query_records.sort()

    # Load the model
    module_name = load_model(config)
    return (training_metadata, query_metadata, query_records,
            strip, nstrip, module_name)


def get_strips(records: List[str]) -> Tuple[int, int]:
    def f(k: str) -> Tuple[int, int]:
        r = os.path.basename(k).rsplit(".", maxsplit=3)[1]
        nums = r.split("of")
        tups = (int(nums[0]), int(nums[1]))
        return tups
    strip_set = {f(k) for k in records}
    if len(strip_set) > 1:
        log.error("TFRecord files can only be from a single strip.")
        sys.exit()
    strip = strip_set.pop()
    return strip


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


def get_traintest_data(
    records_train: List[str],
    records_test: List[str],
    metadata: Training,
    npoints: Optional[int],
    batch_size: int,
    random_seed: int
) -> Tuple[Dict[str, np.ndarray], np.ndarray,
           Dict[str, np.ndarray], np.ndarray]:

    train_dataset = train_data(records_train, metadata, batch_size, epochs=1,
                               take=npoints, random_seed=random_seed)()
    X_tensor, Y_tensor = train_dataset.make_one_shot_iterator().get_next()
    test_dataset = test_data(records_test, metadata, batch_size)()
    Xt_tensor, Yt_tensor = test_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        X, Y = _extract(X_tensor, Y_tensor, sess)
        Xt, Yt = _extract(Xt_tensor, Yt_tensor, sess)
    return X, Y, Xt, Yt


def read_train_record(
    directory: str,
    maxpoints: int,
    random_seed: int = 220,
    batchsize: int = 1000,
) -> np.ndarray:
    training_records, testing_records, metadata = get_training_meta(directory)
    x, y, x_test, y_test = get_traintest_data(
        training_records, testing_records, metadata, maxpoints, batchsize,
        random_seed
    )

    x_con, x_cat, indices, coords = _split(x)
    xt_con, xt_cat, indicest, coordst = _split(x_test)
    return x_con, x_cat, indices, coords, xt_con, xt_cat, indicest, coordst