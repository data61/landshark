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
from landshark.saver import overwrite_model_dir

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
@click.option("--test_batchsize", type=click.IntRange(min=1), default=1000,
              help="Testing batch size")
@click.option("--iterations", type=click.IntRange(min=1), default=None,
              help="number of training/testing iterations.")
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Optional directory containing model checkpoints.")
@click.pass_context
def train(ctx: click.Context, data: str, config: str, epochs: int,
          batchsize: int, test_batchsize: int,
          iterations: Optional[int], checkpoint: Optional[str]) -> None:
    """Train a model specified by a config file."""
    log.info("Ignoring batch-mb option, using specified or default batchsize")
    catching_f = errors.catch_and_exit(train_entrypoint)
    catching_f(data, config, epochs, batchsize, test_batchsize,
               iterations, ctx.obj.gpu, checkpoint)


def train_entrypoint(data: str, config: str, epochs: int, batchsize: int,
                     test_batchsize: int, iterations: Optional[int],
                     gpu: bool, checkpoint_dir: Optional[str]) -> None:
    """Entry point for training function."""
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, data)
    if checkpoint_dir:
        overwrite_model_dir(model_dir, checkpoint_dir)

    training_params = TrainingConfig(epochs, batchsize,
                                     test_batchsize, gpu)
    train_test(training_records, testing_records, metadata, model_dir,
               sys.modules[cf], training_params, iterations)


@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True,
              help="Path to the model file")
@click.option("--checkpoint", type=click.Path(exists=True), required=True,
              help="Path to the trained model checkpoint")
@click.option("--data", type=click.Path(exists=True), required=True,
              help="Path to the query data directory")
@click.pass_context
def predict(
        ctx: click.Context,
        config: str,
        checkpoint: str,
        data: str) -> None:
    """Predict using a learned model."""
    catching_f = errors.catch_and_exit(predict_entrypoint)
    catching_f(config, checkpoint, data, ctx.obj.batchMB, ctx.obj.gpu)


def predict_entrypoint(config: str, checkpoint: str, data: str,
                       batchMB: float, gpu: bool) -> None:
    train_metadata, query_metadata, query_records, strip, nstrips, cf = \
        setup_query(config, data, checkpoint)
    ndims_ord = train_metadata.features.D_ordinal
    ndims_cat = train_metadata.features.D_categorical
    points_per_batch = mb_to_points(batchMB, ndims_ord, ndims_cat,
                                    halfwidth=train_metadata.halfwidth)
    params = QueryConfig(points_per_batch, gpu)
    y_dash_it = predict_fn(checkpoint, sys.modules[cf], train_metadata,
                           query_records, params)
    write_geotiffs(y_dash_it, checkpoint, train_metadata,
                   tag="{}of{}".format(strip, nstrips))


if __name__ == "__main__":
    cli()
