"""Main landshark commands."""

import sys
import logging

import click

from landshark import model
from landshark.tifwrite import write_geotiffs
from landshark.scripts.logger import configure_logging
from landshark.image import strip_image_spec
from landshark.model import TrainingConfig, QueryConfig
from landshark.tfread import setup_training, setup_query, get_strips

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
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, directory)

    # Train
    training_params = TrainingConfig(epochs, batchsize, samples,
                                     test_batchsize, test_samples, gpu)
    model.train_test(training_records, testing_records, metadata, model_dir,
                     sys.modules[cf], training_params)
    return 0


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
    metadata, query_records = setup_query(modeldir, querydir)
    params = QueryConfig(batchsize, samples, [lower, upper], gpu)

    y_dash_it = model.predict(modeldir, metadata, query_records, params)

    strip, nstrips = get_strips(query_records)
    imspec = strip_image_spec(strip, nstrips, metadata.image_spec)
    metadata.image_spec = imspec
    write_geotiffs(y_dash_it, modeldir, metadata, params.percentiles,
                   tag="{}of{}".format(strip, nstrips))
    return 0
