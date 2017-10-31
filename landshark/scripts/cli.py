"""Main landshark commands."""

import logging
import os
from glob import glob
from importlib.util import spec_from_file_location, module_from_spec

import click

from landshark import model
from landshark.hread import ImageFeatures
from landshark.feed import query_data
from landshark.importers.tifwrite import write_geotiffs

log = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
def cli(verbosity: str) -> int:
    """Parse the command line arguments."""
    logging.basicConfig()
    lg = logging.getLogger("")
    lg.setLevel(verbosity)
    return 0


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("name", type=str)
@click.argument("config", type=click.Path(exists=True))
@click.option("--epochs", type=click.IntRange(min=1), default=1,
              help="Epochs between testing the model.")
@click.option("--batchsize", type=click.IntRange(min=1), default=50,
              help="Training batch size")
@click.option("--predict_samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for validating on the"
              " test set.")
def train(directory: str, name: str, config: str, epochs: int, batchsize: int,
          predict_samples: int) -> int:
    """Train a model specified by an input configuration."""

    # Get the data
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))

    # Get metadata for feeding to the model
    metadata_path = os.path.join(directory, "METADATA.bin")
    metadata = model.load_metadata(metadata_path)

    # Load the model
    modspec = spec_from_file_location("config", config)
    cf = module_from_spec(modspec)
    modspec.loader.exec_module(cf)

    # Train
    model.train_test(training_records, testing_records, metadata, name,
                     batchsize, epochs, predict_samples, cf)
    return 0


@cli.command()
@click.argument("featurefile", type=click.Path(exists=True))
@click.argument("modeldir", type=click.Path(exists=True))
@click.argument("metadir", type=click.Path(exists=True))
@click.option("--cache_blocksize", type=int, default=1000)
@click.option("--cache_nblocks", type=int, default=1)
@click.option("--batchsize", type=int, default=100000)
@click.option("--predict_samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for prediction")
def predict(
        featurefile: str,
        modeldir: str,
        metadir: str,
        cache_blocksize: int,
        cache_nblocks: int,
        batchsize: int,
        predict_samples: int
        ) -> int:
    """Predict using a learned model."""
    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)
    metadata = model.load_metadata(os.path.join(metadir, "METADATA.bin"))
    d = query_data(features, batchsize, metadata.halfwidth)
    y_dash_it = model.predict(modeldir, metadata, d, predict_samples)
    write_geotiffs(y_dash_it, modeldir, metadata, features.image_spec)
    return 0
