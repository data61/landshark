"""Landshark importing commands."""

import logging
import os.path
from glob import glob

import click
# mypy type checking
from typing import List

from landshark.importers.tifread import ImageStack
from landshark.hread import ImageFeatures
from landshark.importers.featurewrite import write_datafile
from landshark.importers.shpread import ShapefileTargets
from landshark.scripts.logger import configure_logging
from landshark import feed

log = logging.getLogger(__name__)


# SOME USEFUL PREPROCESSING COMMANDS
# ----------------------------------
# gdal_translate -co "COMPRESS=NONE" src dest


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
def cli(verbosity: str) -> int:
    """Parse the command line arguments."""
    configure_logging(verbosity)
    return 0


def _tifnames(directory: str) -> List[str]:
    names = glob(os.path.join(directory, "*.tif"))
    result = list(filter(lambda x: x.rsplit(".")[1] == "tif", names))
    return result


@cli.command()
@click.option("--categorical", type=click.Path(exists=True))
@click.option("--ordinal", type=click.Path(exists=True))
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--standardise/--no-standardise", default=True,
              help="Standardise the input features")
def tifs(categorical: str, ordinal: str,
         name: str, standardise: bool) -> int:
    """Build a tif stack from a set of input files."""
    out_filename = os.path.join(os.getcwd(), name + ".hdf5")
    cat_tif_filenames = _tifnames(categorical)
    ord_tif_filenames = _tifnames(ordinal)
    stack = ImageStack(cat_tif_filenames, ord_tif_filenames)
    write_datafile(stack, out_filename, standardise)
    return 0


@cli.command()
@click.argument("targets", type=str, nargs=-1)
@click.option("--shapefile", type=click.Path(exists=True), required=True)
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--test_frac", type=float, default=0.1)
@click.option("--random_seed", type=int, default=666)
@click.option("--batchsize", type=int, default=1000)
@click.option("--halfwidth", type=int, default=1)
@click.option("--cache_blocksize", type=int, default=100)
@click.option("--cache_nblocks", type=int, default=10)
def targets(shapefile: str, test_frac: float, random_seed: int,
            features: str, batchsize: int, halfwidth: int,
            cache_blocksize: int, cache_nblocks: int,
            targets: List[str]) -> int:
    """Build training and testing data from shapefile."""
    # file_str = os.path.basename(fname).rsplit(".")[0]
    # tr_filename = os.path.join(os.getcwd(), file_str + "_train" + ".hdf5")
    # ts_filename = os.path.join(os.getcwd(), file_str + "_test" + ".hdf5")
    tr_read = ShapefileTargets(shapefile, targets)
    for r in tr_read.batches():
        print(r)


    # training_targets, testing_targets = train_test_targets(shapefile,
    #                                                        targets,
    #                                                        test_frac,
    #                                                        random_seed)
    feature_obj = ImageFeatures(features, cache_blocksize, cache_nblocks)
    t = feed.training_data(feature_obj, training_targets, halfwidth)
    s = feed.training_data(feature_obj, testing_targets, halfwidth)


    # use lbalpha here
    directory = os.getcwd()
    # to_tfrecords(t, directory, "training", features.ncategories)
    # to_tfrecords(s, directory, "testing", features.ncategories)
    return 0
