"""Main landshark commands."""

import logging
import sys
from typing import NamedTuple, Optional

import click

from landshark import errors
from landshark.model import QueryConfig, TrainingConfig
from landshark.model import predict as predict_fn
from landshark.model import train_test
from landshark.scripts.logger import configure_logging
from landshark.tfread import setup_query, setup_training
from landshark.tifwrite import write_geotiffs
from landshark.util import mb_to_points

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    gpu: bool
    batchMB: float


@click.group()
@click.option("--gpu/--no-gpu", default=False,
              help="Have tensorflow use the GPU")
@click.option("--batch-mb", type=float, default=100,
              help="Approximate size in megabytes of data read per "
              "worker per iteration")
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.pass_context
def cli(ctx: click.Context, gpu: bool, verbosity: str, batch_mb: float) -> int:
    """Train a model and use it to make predictions."""
    ctx.obj = CliArgs(gpu=gpu, batchMB=batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--data", type=click.Path(exists=True), required=True,
              help="The traintest folder containing the data")
@click.option("--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
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
          iterations: Optional[int], learnrate: float) -> None:
    """Train a model specified by  a config file."""
    log.info("Ignoring batch-mb option, using specified or default batchsize")
    catching_f = errors.catch_and_exit(train_entrypoint)
    catching_f(data, config, epochs, batchsize, test_batchsize, samples,
               test_samples, iterations, learnrate, ctx.obj.gpu)


def train_entrypoint(data: str, config: str, epochs: int, batchsize: int,
                     test_batchsize: int, samples: int, test_samples: int,
                     iterations: Optional[int], learnrate: float,
                     gpu: bool) -> None:
    """Entry point for training function."""
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, data)
    training_params = TrainingConfig(epochs, batchsize, samples,
                                     test_batchsize, test_samples, gpu,
                                     learnrate)
    train_test(training_records, testing_records, metadata, model_dir,
               sys.modules[cf], training_params, iterations)


@cli.command()
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Path to the trained model directory")
@click.option("--data", type=click.Path(exists=True), required=True,
              help="Path to the query data directory")
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
        samples: int,
        lower: int,
        upper: int) -> None:
    """Predict using a learned model."""
    catching_f = errors.catch_and_exit(predict_entrypoint)
    catching_f(model, data, ctx.obj.batchMB, samples,
               lower, upper, ctx.obj.gpu)


def predict_entrypoint(model: str, data: str, batchMB: float, samples: int,
                       lower: int, upper: int, gpu: bool) -> None:
    train_metadata, query_metadata, query_records, strip, nstrips = \
        setup_query(model, data)
    percentiles = (float(lower), float(upper))
    ndims_ord = train_metadata.features.D_ordinal
    ndims_cat = train_metadata.features.D_categorical
    points_per_batch = mb_to_points(batchMB, ndims_ord, ndims_cat,
                                    halfwidth=train_metadata.halfwidth)
    params = QueryConfig(points_per_batch, samples, percentiles, gpu)
    y_dash_it = predict_fn(model, train_metadata, query_records, params)
    write_geotiffs(y_dash_it, model, train_metadata,
                   list(params.percentiles),
                   tag="{}of{}".format(strip, nstrips))
