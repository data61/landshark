"""Main landshark commands."""

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
@click.argument("config", type=click.Path(exists=True))
@click.option("--epochs", type=click.IntRange(min=1), default=1,
              help="Epochs between testing the model.")
@click.option("--batchsize", type=click.IntRange(min=1), default=50,
              help="Training batch size")
@click.option("--predict_samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for validating on the"
              " test set.")
def train(directory: str, config: str, epochs: int, batchsize: int,
          predict_samples: int) -> int:
    """Train a model specified by an input configuration."""
    name = os.path.basename(config).rsplit(".")[0]

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
@click.argument("directory", type=click.Path(exists=True))
@click.argument("featurefile", type=click.Path(exists=True))
@click.option("--npoints", type=int, default=1000)
@click.option("--trees", type=int, default=100)
@click.option("--cache_blocksize", type=int, default=1000)
@click.option("--cache_nblocks", type=int, default=1)
def baseline(directory: str, featurefile: str, npoints: int, trees: int,
             cache_blocksize: int, cache_nblocks: int) -> int:
    """Run a random forest model as a baseline for comparison."""

    # Get the data
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))
    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)

    # Get metadata for feeding to the model
    metadata_path = os.path.join(directory, "METADATA.bin")
    metadata = model.load_metadata(metadata_path)

    # Train
    y_it = rf.train_test_predict(training_records, testing_records, metadata,
                                 features, npoints, trees)
    write_geotiffs(y_it, directory, metadata, features.image_spec,
                   tag="baseline", lower=0, upper=0)
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
@click.option("--lower", type=click.IntRange(min=0, max=100), default=10,
              help="Lower percentile of the predictive density to output")
@click.option("--upper", type=click.IntRange(min=0, max=100), default=90,
              help="Upper percentile of the predictive density to output")
def predict(
        featurefile: str,
        modeldir: str,
        metadir: str,
        cache_blocksize: int,
        cache_nblocks: int,
        batchsize: int,
        predict_samples: int,
        lower: int,
        upper: int
        ) -> int:
    """Predict using a learned model."""
    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)
    metadata = model.load_metadata(os.path.join(metadir, "METADATA.bin"))
    d = query_data(features, batchsize, metadata.halfwidth)
    y_dash_it = model.predict(modeldir, metadata, d, predict_samples, lower,
                              upper)
    write_geotiffs(y_dash_it, modeldir, metadata, features.image_spec, lower,
                   upper)
    return 0
