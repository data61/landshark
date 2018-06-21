"""Main landshark commands."""

import logging
import os
from shutil import copyfile

from typing import Optional
import click

from landshark import skmodel
from landshark.tifwrite import write_geotiffs
from landshark.scripts.logger import configure_logging
from landshark.tfread import setup_training, setup_query, load_model

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
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=click.IntRange(min=1), default=1000,
              help="Training batch size")
@click.option("--maxpoints", type=int, default=None)
@click.option("--random_seed", type=int, default=666)
def train(data: str, config: str, batchsize: int,
          maxpoints: Optional[int], random_seed: int) -> int:
    """Train a model specified by an input configuration."""
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, data)

    # copy the model spec to the model dir
    copyfile(config, os.path.join(model_dir, "config.py"))
    skmodel.train_test(cf, training_records, testing_records,
                       metadata, model_dir, maxpoints,
                       batchsize, random_seed)
    return 0


@cli.command()
@click.option("--model", type=click.Path(exists=True), required=True)
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=100000)
@click.option("--lower", type=click.IntRange(min=0, max=100), default=10,
              help="Lower percentile of the predictive density to output")
@click.option("--upper", type=click.IntRange(min=0, max=100), default=90,
              help="Upper percentile of the predictive density to output")
def predict(model: str, data: str, batchsize: int,
            lower: int, upper: int) -> int:
    """Predict using a learned model."""
    train_metadata, query_metadata, query_records, strip, nstrips = \
        setup_query(model, data)
    percentiles = (float(lower), float(upper))
    load_model(os.path.join(model, "config.py"))
    y_dash_it = skmodel.predict(model, train_metadata, query_records,
                                batchsize, percentiles)
    write_geotiffs(y_dash_it, model, train_metadata,
                   list(percentiles), tag="{}of{}".format(strip, nstrips))
    return 0
