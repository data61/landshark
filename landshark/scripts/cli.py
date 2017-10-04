"""Main landshark commands."""

import logging

import tables
import click

from landshark.hread import ImageFeatures, Targets, training_data, query_data
from landshark import models

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
@click.argument("featurefile", type=click.Path(exists=True))
@click.argument("targetfile", type=click.Path(exists=True))
@click.argument("name", type=str)
@click.option("--model", type=click.Choice(["Aboleth", "RF", "Linear"]),
              default="Linear")
@click.option("--batchsize", type=int, default=1000)
@click.option("--halfwidth", type=int, default=1)
@click.option("--cache_blocksize", type=int, default=100)
@click.option("--cache_nblocks", type=int, default=10)
@click.option("--target", type=str, required=True)
def train(featurefile: str, targetfile: str, name: str, model: str, batchsize: int,
          cache_blocksize: int, cache_nblocks: int, halfwidth: int, target) -> int:
    """Learn a model."""

    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)
    targets = Targets(targetfile, target)
    t = training_data(features, targets, batchsize, halfwidth)
    m = models.train(t)
    models.write(m, halfwidth, target, name)


@cli.command()
@click.argument("featurefile", type=click.Path(exists=True))
@click.argument("modelfile", type=click.Path(exists=True))
@click.option("--cache_blocksize", type=int, default=100)
@click.option("--cache_nblocks", type=int, default=10)
@click.option("--batchsize", type=int, default=1000)
def predict(featurefile, modelfile, cache_blocksize, cache_nblocks,
            batchsize) -> int:
    """Predict using a learned model."""
    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)
    m = models.load(modelfile)
    model = m.skmodel
    halfwidth = m.halfwidth
    d = query_data(features, batchsize, halfwidth)
    y_dash = models.predict(model, d)
    models.show(y_dash, features.image_spec)
    return 0
