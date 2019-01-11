"""Import data from tensorflow format."""

import logging
import os
import sys
from glob import glob
from importlib.util import module_from_spec, spec_from_file_location
from typing import List, Tuple

from landshark.metadata import FeatureSet, Training

log = logging.getLogger(__name__)


def _load_config(module_name: str, path: str) -> None:
    # Load the model
    modspec = spec_from_file_location(module_name, path)
    cf = module_from_spec(modspec)
    if not modspec.loader:
        raise RuntimeError("Could not load configuration module")
    modspec.loader.exec_module(cf)
    # needed for pickling??
    sys.modules[module_name] = cf


def load_model(config_file: str) -> str:
    module_name = "userconfig"
    _load_config(module_name, config_file)
    return module_name


def setup_training(config: str, directory: str) -> \
        Tuple[List[str], List[str], Training, str, str]:
    # Get the data
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))

    # Get metadata for feeding to the model
    metadata = Training.load(directory)

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


def setup_query(config: str, querydir: str, checkpoint: str) \
        -> Tuple[Training, FeatureSet, List[str], int, int, str]:
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
