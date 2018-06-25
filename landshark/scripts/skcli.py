"""Main landshark commands."""

import logging
import os
from shutil import copyfile

from typing import Optional, NamedTuple
import click

from landshark import skmodel
from landshark.tifwrite import write_geotiffs
from landshark.scripts.logger import configure_logging
from landshark.tfread import setup_training, setup_query, load_model
from landshark.util import mb_to_points
from landshark import errors

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    batchMB: int


@click.group()
@click.option("--batch-mb", type=int, default=100)
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.pass_context
def cli(ctx: click.Context, verbosity: str, batch_mb: int) -> int:
    """Parse the command line arguments."""
    ctx.obj = CliArgs(batchMB=batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--maxpoints", type=int, default=None)
@click.option("--random_seed", type=int, default=666)
@click.pass_context
def train(ctx: click.Context, data: str, config: str,
          maxpoints: Optional[int], random_seed: int) -> None:
    """Train a model specified by an input configuration."""
    catching_f = errors.catch_and_exit(train_entrypoint)
    catching_f(data, config, maxpoints, random_seed, ctx.obj.batchMB)


def train_entrypoint(data: str, config: str, maxpoints: Optional[int],
                     random_seed: int, batchMB: int) -> None:
    """Entry point for sklearn model training."""
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, data)

    ndims_ord = metadata.features.ordinal.D \
        if metadata.features.ordinal else 0
    ndims_cat = metadata.features.categorical.D \
        if metadata.features.categorical else 0
    batchsize = mb_to_points(batchMB, ndims_ord, ndims_cat,
                             halfwidth=metadata.halfwidth)
    # copy the model spec to the model dir
    copyfile(config, os.path.join(model_dir, "config.py"))
    skmodel.train_test(cf, training_records, testing_records,
                       metadata, model_dir, maxpoints,
                       batchsize, random_seed)


@cli.command()
@click.option("--model", type=click.Path(exists=True), required=True)
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--lower", type=click.IntRange(min=0, max=100), default=10,
              help="Lower percentile of the predictive density to output")
@click.option("--upper", type=click.IntRange(min=0, max=100), default=90,
              help="Upper percentile of the predictive density to output")
@click.pass_context
def predict(ctx: click.Context, model: str, data: str, lower: int,
            upper: int) -> None:
    """Predict using a learned model."""
    catching_f = errors.catch_and_exit(predict_entrypoint)
    catching_f(model, data, lower, upper, ctx.obj.batchMB)


def predict_entrypoint(model: str, data: str, lower: int, upper: int,
                       batchMB: int) -> None:
    """Entry point for prediction with sklearn."""
    train_metadata, query_metadata, query_records, strip, nstrips = \
        setup_query(model, data)
    percentiles = (float(lower), float(upper))
    ndims_ord = train_metadata.features.ordinal.D \
        if train_metadata.features.ordinal else 0
    ndims_cat = train_metadata.features.categorical.D \
        if train_metadata.features.categorical else 0
    points_per_batch = mb_to_points(batchMB, ndims_ord, ndims_cat,
                                    halfwidth=train_metadata.halfwidth)
    load_model(os.path.join(model, "config.py"))
    y_dash_it = skmodel.predict(model, train_metadata, query_records,
                                points_per_batch, percentiles)
    write_geotiffs(y_dash_it, model, train_metadata,
                   list(percentiles), tag="{}of{}".format(strip, nstrips))
