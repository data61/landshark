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
from landshark.importers import tfwrite
from landshark.importers import metadata as mt
from landshark.scripts.logger import configure_logging
from landshark import feed
from landshark.image import indices_strip

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
    if directory is not None:
        file_types = ('tif', 'gtif')
        names = []
        for t in file_types:
            glob_pattern = os.path.join(directory, "**", "*.{}".format(t))
            names.extend(glob(glob_pattern, recursive=True))
    else:
        names = None
    return names


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
    out_filename = os.path.join(os.getcwd(), name + "_features.hdf5")

    ord_stack, cat_stack = None, None

    if ordinal:
        ord_tif_filenames = _tifnames(ordinal)
        ord_stack = ImageStack(ord_tif_filenames, "ordinal")
    if categorical:
        cat_tif_filenames = _tifnames(categorical)
        cat_stack = ImageStack(cat_tif_filenames, "categorical")

    write_datafile(ord_stack, cat_stack, out_filename, standardise)
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
@click.option("--name", type=str, required=True)
@click.option("--every", type=int, default=1)
@click.option("--categorical", is_flag=True)
def targets(shapefile: str, test_frac: float, random_seed: int,
            features: str, batchsize: int, halfwidth: int,
            cache_blocksize: int, cache_nblocks: int,
            targets: List[str],
            name: str,
            every: int,
            categorical: bool) -> int:
    """Build training and testing data from shapefile."""
    log.info("Loading shapefile targets")
    target_obj = ShapefileTargets(shapefile, targets, batchsize, every,
                                  categorical)
    log.info("Loading image feature stack")
    feature_obj = ImageFeatures(features, cache_blocksize, cache_nblocks)
    target_it = target_obj.batches()
    training_it = feed.training_data(target_it, feature_obj, halfwidth)
    directory = os.path.join(os.getcwd(), name + "_trainingdata")
    log.info("Writing training data to tfrecords")
    n_train = tfwrite.training(training_it, directory, test_frac, random_seed)

    log.info("Writing metadata")
    metadata = mt.from_data(feature_obj, target_obj, halfwidth, n_train)
    mt.write_metadata(directory, metadata)
    return 0


@cli.command()
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--random_seed", type=int, default=666)
@click.option("--batchsize", type=int, default=1000)
@click.option("--halfwidth", type=int, default=1)
@click.option("--cache_blocksize", type=int, default=100)
@click.option("--cache_nblocks", type=int, default=10)
@click.argument("strip", type=int)
@click.argument("totalstrips", type=int)
def queries(random_seed: int,
            features: str, batchsize: int, halfwidth: int,
            cache_blocksize: int, cache_nblocks: int,
            strip: int, totalstrips: int) -> int:
    """Grab a chunk for prediction."""
    log.info("Loading image feature stack")

    dirname = os.path.basename(features).rsplit(".")[0] + \
        "_query{}of{}".format(strip, totalstrips)
    directory = os.path.join(os.getcwd(), dirname)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    feature_obj = ImageFeatures(features, cache_blocksize, cache_nblocks)
    indices_it = indices_strip(feature_obj.image_spec, strip, totalstrips,
                               batchsize)
    data_it = feed.query_data(indices_it, feature_obj, halfwidth)
    tag = "query.{}of{}".format(strip, totalstrips)
    tfwrite.query(data_it, directory, tag=tag)
    return 0
