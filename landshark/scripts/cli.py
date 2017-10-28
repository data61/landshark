"""Main landshark commands."""

from glob import glob
import logging
import os

import click

from landshark import model

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
@click.argument("directory", type=click.Path(exists=True))
@click.argument("name", type=str)
def train(directory: str,
          name: str) -> int:
    """Do stuff."""
    test_dir = os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "*.tfrecord"))
    metadata_path = os.path.join(directory, "METADATA.bin")
    metadata = model.load_metadata(metadata_path)
    model.train_test(training_records, testing_records, metadata, name)
    return 0


@cli.command()
@click.argument("featurefile", type=click.Path(exists=True))
@click.argument("modeldir", type=click.Path(exists=True))
@click.argument("metadir", type=click.Path(exists=True))
@click.option("--cache_blocksize", type=int, default=1000)
@click.option("--cache_nblocks", type=int, default=1)
@click.option("--batchsize", type=int, default=100000)
def predict(
        featurefile: str,
        modeldir: str,
        metadir: str,
        cache_blocksize: int,
        cache_nblocks: int,
        batchsize: int
        ) -> int:
    """Predict using a learned model."""
    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)
    metadata = model.load_metadata(os.path.join(metadir, "METADATA.bin"))
    d = query_data(features, batchsize, metadata.halfwidth)
    y_dash = model.predict(modeldir, metadata, d)
    model.show(y_dash, features.image_spec)
    return 0
