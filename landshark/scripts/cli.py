"""Main landshark commands."""

from glob import glob
import logging
import os

import click

from landshark.hread import ImageFeatures, Targets
from landshark.feed import training_data, query_data
from landshark.export import to_tfrecords
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
@click.argument("featurefile", type=click.Path(exists=True))
@click.argument("trainingfile", type=click.Path(exists=True))
@click.argument("testingfile", type=click.Path(exists=True))
@click.option("--batchsize", type=int, default=1000)
@click.option("--halfwidth", type=int, default=1)
@click.option("--cache_blocksize", type=int, default=100)
@click.option("--cache_nblocks", type=int, default=10)
@click.option("--target", type=str, required=True)
def export(
        featurefile: str,
        trainingfile: str,
        testingfile: str,
        batchsize: int,
        cache_blocksize: int,
        cache_nblocks: int,
        halfwidth: int,
        target: str
        ) -> int:
    """Export data to tfrecord files."""
    features = ImageFeatures(featurefile, cache_blocksize, cache_nblocks)
    training_targets = Targets(trainingfile, target)
    testing_targets = Targets(testingfile, target)
    t = training_data(features, training_targets, batchsize, halfwidth)
    s = training_data(features, testing_targets, batchsize, halfwidth)
    directory = os.getcwd()
    to_tfrecords(t, directory, "training")
    to_tfrecords(s, directory, "testing")
    return 0


@cli.command()
@click.argument("trainingdir", type=click.Path(exists=True))
@click.argument("testingdir", type=click.Path(exists=True))
@click.argument("name", type=str)
def train(
        trainingdir: str,
        testingdir: str,
        name: str) -> int:
    """Do stuff."""
    training_records = glob(os.path.join(trainingdir, "*.tfrecord"))
    testing_records = glob(os.path.join(testingdir, "*.tfrecord"))
    train_metadata_path = os.path.join(trainingdir, "METADATA.bin")
    test_metadata_path = os.path.join(testingdir, "METADATA.bin")
    train_metadata = model.load_metadata(train_metadata_path)
    test_metadata = model.load_metadata(test_metadata_path)
    model.train_test(training_records, testing_records, train_metadata,
                     test_metadata, name)
    return 0


@cli.command()
@click.argument("featurefile", type=click.Path(exists=True))
@click.argument("modeldir", type=click.Path(exists=True))
@click.argument("metadir", type=click.Path(exists=True))
@click.option("--cache_blocksize", type=int, default=100)
@click.option("--cache_nblocks", type=int, default=10)
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
    y_dash = model.predict(modeldir, d)
    model.show(y_dash, features.image_spec)
    return 0
