"""Main landshark commands."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from shutil import copyfile
from typing import NamedTuple, Optional

import click

from landshark import __version__, errors, skmodel
from landshark.model import setup_query, setup_training
from landshark.scripts.logger import configure_logging
from landshark.tifwrite import write_geotiffs
from landshark.util import mb_to_points

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    batchMB: float


@click.group()
@click.version_option(version=__version__)
@click.option("--batch-mb", type=float, default=100,
              help="Approximate size in megabytes of data read per "
              "worker per iteration")
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.pass_context
def cli(ctx: click.Context, verbosity: str, batch_mb: float) -> int:
    """Train and predict using scikit-learn style models (in memory!)."""
    ctx.obj = CliArgs(batchMB=batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--data", type=click.Path(exists=True), required=True,
              help="The traintest folder containing the data")
@click.option("--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
@click.option("--maxpoints", type=int, default=None,
              help="Limit the number of training points "
              "supplied to the sklearn model")
@click.option("--random_seed", type=int, default=666,
              help="Random state supplied to sklearn for reproducibility")
@click.pass_context
def train(ctx: click.Context,
          data: str,
          config: str,
          maxpoints: Optional[int],
          random_seed: int
          ) -> None:
    """Train a model specified by an sklearn input configuration."""
    catching_f = errors.catch_and_exit(train_entrypoint)
    catching_f(data, config, maxpoints, random_seed, ctx.obj.batchMB)


def train_entrypoint(data: str,
                     config: str,
                     maxpoints: Optional[int],
                     random_seed: int,
                     batchMB: float
                     ) -> None:
    """Entry point for sklearn model training."""
    metadata, training_records, testing_records, model_dir, cf = \
        setup_training(config, data)

    ndims_con = len(metadata.features.continuous) \
        if metadata.features.continuous else 0
    ndims_cat = len(metadata.features.categorical) \
        if metadata.features.categorical else 0
    batchsize = mb_to_points(batchMB, ndims_con, ndims_cat,
                             metadata.features.halfwidth)
    # copy the model spec to the model dir
    copyfile(config, os.path.join(model_dir, "config.py"))
    skmodel.train_test(cf, training_records, testing_records,
                       metadata, model_dir, maxpoints,
                       batchsize, random_seed)


@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True,
              help="Path to the model file")
@click.option("--checkpoint", type=click.Path(exists=True), required=True,
              help="Path to the trained model checkpoint")
@click.option("--data", type=click.Path(exists=True), required=True,
              help="Path to the query data directory")
@click.pass_context
def predict(ctx: click.Context,
            config: str,
            checkpoint: str,
            data: str
            ) -> None:
    """Predict using a learned model."""
    catching_f = errors.catch_and_exit(predict_entrypoint)
    catching_f(config, checkpoint, data, ctx.obj.batchMB)


def predict_entrypoint(config: str,
                       checkpoint: str,
                       data: str,
                       batchMB: float
                       ) -> None:
    """Entry point for prediction with sklearn."""
    train_metadata, query_metadata, query_records, strip, nstrips, _ = \
        setup_query(config, data, checkpoint)
    ndims_con = len(train_metadata.features.continuous) \
        if train_metadata.features.continuous else 0
    ndims_cat = len(train_metadata.features.categorical) \
        if train_metadata.features.categorical else 0
    points_per_batch = mb_to_points(batchMB, ndims_con, ndims_cat,
                                    train_metadata.features.halfwidth)

    y_dash_it = skmodel.predict(checkpoint, train_metadata, query_records,
                                points_per_batch)
    write_geotiffs(y_dash_it, checkpoint, query_metadata.image,
                   tag="{}of{}".format(strip, nstrips))


if __name__ == "__main__":
    cli()
