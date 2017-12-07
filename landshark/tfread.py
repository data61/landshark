from importlib.util import spec_from_file_location, module_from_spec
from glob import glob
import os
import logging
import sys
from landshark.metadata import load_metadata, write_metadata

log = logging.getLogger(__name__)


def _load_config(module_name, path):
    # Load the model
    modspec = spec_from_file_location(module_name, path)
    cf = module_from_spec(modspec)
    modspec.loader.exec_module(cf)
    # needed for pickling??
    sys.modules[module_name] = cf


def setup_training(config, directory):
    name = os.path.basename(config).rsplit(".")[0] + "_model"

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
    module_name = "userconfig"
    _load_config(module_name, config)

    return training_records, testing_records, metadata, model_dir, module_name


def setup_query(modeldir, querydir):
    metadata = load_metadata(os.path.join(modeldir, "METADATA.bin"))
    query_records = glob(os.path.join(querydir, "*.tfrecord"))
    query_records.sort()
    config_file = os.path.join(modeldir, "config.py")
    _load_config("userconfig", config_file)
    return metadata, query_records

def get_strips(records):
    def f(k):
        r = os.path.basename(k).rsplit(".", maxsplit=3)[1]
        tups = tuple(int(i) for i in r.split("of"))
        return tups
    strip_set = set(f(k) for k in records)
    if len(strip_set) > 1:
        log.error("TFRecord files can only be from a single strip.")
        sys.exit()
    strip = strip_set.pop()
    return strip

