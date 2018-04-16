"""Dump data to hdf5 for external tools."""
import logging
from multiprocessing import cpu_count
import os

import click

from landshark.scripts.logger import configure_logging
from landshark.trainingdata import setup_training
from landshark.dump import to_hdf5
from landshark.metadata import from_files

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
@click.argument("features", type=click.Path(exists=True))
@click.argument("targets", type=click.Path(exists=True))
@click.option("--folds", type=click.IntRange(2, None), default=10)
@click.option("--halfwidth", type=click.IntRange(0, None), default=1)
@click.option("--nworkers", type=click.IntRange(0, None), default=cpu_count())
@click.option("--batchsize", type=click.IntRange(1, None), default=100)
@click.option("--random_seed", type=int, default=666)
def trainingdata(features: str, targets: str, folds: int,
                 halfwidth: int, batchsize: int, nworkers: int,
                 random_seed: int) -> int:
    """Get training data."""

    testfold = 1  # ignored really -- all data written in and fold assignments
    tinfo = setup_training(features, targets, folds, random_seed, halfwidth)
    outfile_name = os.path.join(os.getcwd(), tinfo.name + "_traintest.hdf5")
    n_train = len(tinfo.target_src) - tinfo.folds.counts[testfold]
    metadata = from_files(features, targets, tinfo.image_spec,
                          halfwidth, n_train, folds, testfold)
    to_hdf5(tinfo, metadata, outfile_name, batchsize, nworkers)
    log.info("Training dump complete")
    return 0

