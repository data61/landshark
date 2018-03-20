"""Main landshark commands."""

import logging
import os
from shutil import copyfile

import click

from landshark import skmodel
from landshark.tifwrite import write_geotiffs
from landshark.scripts.logger import configure_logging
from landshark.image import strip_image_spec
from landshark.tfread import setup_training, get_strips, setup_query, load_model
from landshark.metadata import TrainingMetadata

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
@click.option("--batchsize", type=click.IntRange(min=1), default=1000,
              help="Training batch size")
@click.option("--maxpoints", type=int, default=2000)
@click.option("--random_seed", type=int, default=666)
def train(directory: str, config: str, batchsize: int, maxpoints: int,
          random_seed: int) -> int:
    """Train a model specified by an input configuration."""
    training_records, testing_records, metadata, model_dir, cf = \
        setup_training(config, directory)

    # copy the model spec to the model dir
    copyfile(config, os.path.join(model_dir, "config.py"))
    skmodel.train_test(cf, training_records, testing_records,
                       metadata, model_dir, maxpoints,
                       batchsize, random_seed)
    return 0


@cli.command()
@click.argument("modeldir", type=click.Path(exists=True))
@click.argument("querydir", type=click.Path(exists=True))
@click.option("--batchsize", type=int, default=100000)
def predict(modeldir: str, querydir: str, batchsize: int) -> int:
    """Predict using a learned model."""

    load_model(os.path.join(modeldir, "config.py"))
    metadata, query_records = setup_query(modeldir, querydir)
    y_dash_it = skmodel.predict(modeldir, metadata, query_records, batchsize)
    strip, nstrips = get_strips(query_records)
    strip_imspec = strip_image_spec(strip, nstrips, metadata.image_spec)
    md_dict = metadata._asdict()
    md_dict["image_spec"] = strip_imspec
    strip_metadata = TrainingMetadata(**md_dict)
    write_geotiffs(y_dash_it, modeldir, strip_metadata, percentiles=None,
                   tag="{}of{}".format(strip, nstrips))
    return 0
