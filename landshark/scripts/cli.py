"""Main landshark commands."""
import sys
import logging
from typing import Optional, NamedTuple

import click

from landshark.tifwrite import write_geotiffs
from landshark.scripts.logger import configure_logging
from landshark.model import TrainingConfig, QueryConfig, predict, train_test
from landshark.tfread import setup_training, setup_query

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    gpu: bool


@click.group()
@click.option("--gpu/--no-gpu", default=False)
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.pass_context
def cli(ctx: click.Context, gpu: bool, verbosity: str) -> int:
    """Parse the command line arguments."""
    ctx.obj = CliArgs(gpu=gpu)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--epochs", type=click.IntRange(min=1), default=1,
              help="Epochs between testing the model.")
@click.option("--batchsize", type=click.IntRange(min=1), default=1000,
              help="Training batch size")
@click.option("--samples", type=click.IntRange(min=1), default=5,
              help="Number of times to sample the model for training.")
@click.option("--test_samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for validating on the"
              " test set.")
@click.option("--learnrate", type=float, default=0.01,
              help="Learning rate to pass to ADAM optimiser")
@click.option("--test_batchsize", type=click.IntRange(min=1), default=1000,
              help="Testing batch size")
@click.option("--iterations", type=click.IntRange(min=1), default=None,
              help="number of training/testing iterations.")
@click.pass_context
def train(ctx: click.Context, data: str, config: str, epochs: int,
          batchsize: int, test_batchsize: int, samples: int, test_samples: int,
          gpu: bool, iterations: Optional[int], learnrate: float) -> int:
    """Train a model specified by an input configuration."""
    gpu = ctx.obj.gpu
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, data)

    # Train
    training_params = TrainingConfig(epochs, batchsize, samples,
                                     test_batchsize, test_samples, gpu,
                                     learnrate)
    train_test(training_records, testing_records, metadata, model_dir,
               sys.modules[cf], training_params, iterations)
    return 0


@cli.command()
@click.option("--model", type=click.Path(exists=True), required=True)
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=100000)
@click.option("--samples", type=click.IntRange(min=1), default=20,
              help="Number of times to sample the model for prediction")
@click.option("--lower", type=click.IntRange(min=0, max=100), default=10,
              help="Lower percentile of the predictive density to output")
@click.option("--upper", type=click.IntRange(min=0, max=100), default=90,
              help="Upper percentile of the predictive density to output")
@click.pass_context
def predict(
        ctx: click.Context,
        model: str,
        data: str,
        batchsize: int,
        samples: int,
        lower: int,
        upper: int) -> int:
    """Predict using a learned model."""
    gpu = ctx.obj.gpu
    train_metadata, query_metadata, query_records, strip, nstrips = \
        setup_query(model, data)
    percentiles = (float(lower), float(upper))
    params = QueryConfig(batchsize, samples, percentiles, gpu)
    y_dash_it = predict(model, train_metadata, query_records, params)
    write_geotiffs(y_dash_it, model, train_metadata,
                   list(params.percentiles),
                   tag="{}of{}".format(strip, nstrips))
    return 0
