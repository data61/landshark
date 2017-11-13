"""Main landshark commands."""

import sys
import logging
import os
from glob import glob
from importlib.util import spec_from_file_location, module_from_spec

import click

from landshark import model
from landshark.hread import ImageFeatures
from landshark.feed import query_data
from landshark import rf
from landshark.importers.tifwrite import write_geotiffs
from landshark.scripts.logger import configure_logging
from landshark.image import strip_image_spec
from landshark.importers.metadata import write_metadata, load_metadata
from landshark.model import TrainingConfig, QueryConfig

log = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
def cli(verbosity: str) -> int:
    """Parse the command line arguments."""
    configure_logging(verbosity)
    return 0

@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--batchsize", type=click.IntRange(min=1), default=1000,
              help="Training batch size")
@click.option("--maxpoints", type=int, default=2000)
@click.option("--trees", type=int, default=100)
@click.option("--random_seed", type=int, default=666)
def baseline(directory: str, batchsize: int, maxpoints: int,
             trees: int, random_seed: int) -> int:
    """Train a model specified by an input configuration."""
    name = os.path.basename(directory) + "_baseline"

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

    # Train
    print('calling train')
    rf.train_test(training_records, testing_records, metadata, model_dir,
                  maxpoints, trees, batchsize, random_seed)
    print('called train')
    return 0


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("config", type=click.Path(exists=True))
@click.option("--epochs", type=click.IntRange(min=1), default=1,
              help="Epochs between testing the model.")
@click.option("--batchsize", type=click.IntRange(min=1), default=1000,
              help="Training batch size")
@click.option("--samples", type=click.IntRange(min=1), default=5,
              help="Number of times to sample the model for training.")
@click.option("--test_samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for validating on the"
              " test set.")
@click.option("--test_batchsize", type=click.IntRange(min=1), default=1000,
              help="Testing batch size")
@click.option("--gpu/--no-gpu", default=False)
def train(directory: str, config: str, epochs: int, batchsize: int,
          test_batchsize: int, samples: int, test_samples: int,
          gpu: bool) -> int:
    """Train a model specified by an input configuration."""
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
    modspec = spec_from_file_location("config", config)
    cf = module_from_spec(modspec)
    modspec.loader.exec_module(cf)

    # Train
    training_params = TrainingConfig(epochs, batchsize, samples,
                                     test_batchsize, test_samples, gpu)
    model.train_test(training_records, testing_records, metadata, model_dir,
                     cf, training_params)
    return 0


def _get_strips(records):
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


@cli.command()
@click.argument("modeldir", type=click.Path(exists=True))
@click.argument("querydir", type=click.Path(exists=True))
@click.option("--batchsize", type=int, default=100000)
@click.option("--gpu/--no-gpu", default=False)
@click.option("--samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for prediction")
@click.option("--lower", type=click.IntRange(min=0, max=100), default=10,
              help="Lower percentile of the predictive density to output")
@click.option("--upper", type=click.IntRange(min=0, max=100), default=90,
              help="Upper percentile of the predictive density to output")
def predict(
        modeldir: str,
        querydir: str,
        batchsize: int,
        samples: int,
        lower: int,
        upper: int,
        gpu: bool) -> int:
    """Predict using a learned model."""
    metadata = load_metadata(os.path.join(modeldir, "METADATA.bin"))
    query_records = glob(os.path.join(querydir, "*.tfrecord"))
    params = QueryConfig(batchsize, samples, [lower, upper], gpu)
    y_dash_it = model.predict(modeldir, metadata, query_records, params)
    strip, nstrips = _get_strips(query_records)
    imspec = strip_image_spec(strip, nstrips, metadata.image_spec)
    metadata.image_spec = imspec
    write_geotiffs(y_dash_it, modeldir, metadata, params.percentiles,
                   tag="{}of{}".format(strip, nstrips))
    return 0


