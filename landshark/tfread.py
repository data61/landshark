from importlib.util import spec_from_file_location, module_from_spec
from glob import glob
import os
import logging
import sys
from landshark.metadata import load_metadata, write_metadata, TrainingMetadata
from typing import Tuple, List

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
        Tuple[List[str], List[str], TrainingMetadata, str, str]:
    name = os.path.basename(config).rsplit(".")[0] + "_model"
    # name = os.path.basename(directory) + "_" + \
    #     os.path.basename(config).rsplit(".")[0] + "_model"

    # Get the data
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))

    # Get metadata for feeding to the model
    metadata_path = os.path.join(directory, "METADATA.bin")
    metadata = load_metadata(metadata_path)

    # Write the metadata
    model_dir = os.path.join(os.getcwd(), name)
    try:
        os.makedirs(model_dir)
    except FileExistsError:
        pass
    write_metadata(model_dir, metadata)

    # Load the model
    module_name = load_model(config)

    return training_records, testing_records, metadata, model_dir, module_name


def setup_query(modeldir: str, querydir: str) \
        -> Tuple[TrainingMetadata, List[str]]:
    metadata = load_metadata(os.path.join(modeldir, "METADATA.bin"))
    query_records = glob(os.path.join(querydir, "*.tfrecord"))
    query_records.sort()
    return metadata, query_records


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
