"""Landshark importing commands."""

import logging
import os.path
from glob import glob
from multiprocessing import cpu_count, Pool

import tables
import click
# mypy type checking
from typing import List

from landshark.tifread import shared_image_spec, OrdinalStackArraySource, \
    CategoricalStackArraySource

# from landshark.hread import ImageFeatures
from landshark.featurewrite import write_imagespec, write_ordinal, \
    write_categorical, write_coordinates
from landshark.shpread import OrdinalShpArraySource,  \
    GenericShpArraySource, CoordinateShpArraySource
# from landshark.importers import tfwrite
# from landshark.importers import metadata as mt
from landshark.scripts.logger import configure_logging
# from landshark import feed
from landshark.hread import read_image_spec
from landshark.trainingdata import write_trainingdata, write_querydata
from landshark.metadata import from_files, write_metadata
# from landshark.image import strip_image_spec

log = logging.getLogger(__name__)


class DummyPool:
    def __init__(self):
        pass

    def imap(self, f, x):
        return map(f, x)

    def terminate(self):
        pass


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
@click.option("--batchsize", type=int, default=1000)
@click.option("--categorical", type=click.Path(exists=True))
@click.option("--ordinal", type=click.Path(exists=True))
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--nworkers", type=int, default=cpu_count())
def tifs(categorical: str, ordinal: str,
         name: str, nworkers: int, batchsize: int) -> int:
    """Build a tif stack from a set of input files."""
    pool = Pool(nworkers) if nworkers > 1 else DummyPool()
    log.info("Using {} worker processes".format(nworkers))
    out_filename = os.path.join(os.getcwd(), name + "_features.hdf5")
    ord_filenames = _tifnames(ordinal) if ordinal else []
    cat_filenames = _tifnames(categorical) if categorical else []
    all_filenames = ord_filenames + cat_filenames
    spec = shared_image_spec(all_filenames)

    with tables.open_file(out_filename, mode="w", title=name) as h5file:

        write_imagespec(spec, h5file)

        if ordinal:
            ord_source = OrdinalStackArraySource(spec, ord_filenames)
            write_ordinal(ord_source, h5file, batchsize, pool)

        if categorical:
            cat_source = CategoricalStackArraySource(spec, cat_filenames)
            write_categorical(cat_source, h5file, batchsize, pool)

    return 0

@cli.command()
@click.argument("targets", type=str, nargs=-1)
@click.option("--shapefile", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=1000)
@click.option("--name", type=str, required=True)
@click.option("--every", type=int, default=1)
@click.option("--categorical", is_flag=True)
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--random_seed", type=int, default=666)
def targets(shapefile: str, batchsize: int, targets: List[str], name: str,
            every: int, categorical: bool,
            nworkers: int, random_seed: int) -> int:
    """Build target file from shapefile."""
    log.info("Loading shapefile targets")
    pool = Pool(nworkers) if nworkers > 1 else DummyPool()
    log.info("Using {} worker processes".format(nworkers))
    out_filename = os.path.join(os.getcwd(), name + "_targets.hdf5")
    with tables.open_file(out_filename, mode="w", title=name) as h5file:

        coord_src = CoordinateShpArraySource(shapefile, random_seed)
        write_coordinates(coord_src, h5file, batchsize)

        if categorical:
            cat_source = GenericShpArraySource(shapefile, targets,
                                               random_seed)
            write_categorical(cat_source, h5file, batchsize, pool)
        else:
            ord_source = OrdinalShpArraySource(shapefile, targets, random_seed)
            write_ordinal(ord_source, h5file, batchsize, pool)
    return 0


@cli.command()
@click.argument("features", type=click.Path(exists=True))
@click.argument("targets", type=click.Path(exists=True))
@click.option("--folds", type=click.IntRange(2, None), default=10)
@click.option("--testfold", type=click.IntRange(1, None), default=1)
@click.option("--halfwidth", type=click.IntRange(0, None), default=1)
@click.option("--nworkers", type=click.IntRange(1, None), default=cpu_count())
@click.option("--batchsize", type=click.IntRange(1, None), default=1000)
@click.option("--random_seed", type=int, default=666)
def trainingdata(features: str, targets: str, testfold: int,
                 folds: int, halfwidth: int, batchsize: int, nworkers: int,
                 random_seed: int) -> int:
    """Get training data."""
    pool = Pool(nworkers) if nworkers > 1 else DummyPool()
    log.info("Using {} worker processes".format(nworkers))
    name = os.path.basename(features).rsplit("_features.")[0] + "-" + \
        os.path.basename(targets).rsplit("_targets.")[0]
    directory = os.path.join(os.getcwd(), name +
                             "_traintest{}of{}".format(testfold, folds))

    image_spec = read_image_spec(features)
    n_train = write_trainingdata(features, targets, image_spec, batchsize,
                                 halfwidth, pool, directory,
                                 testfold, folds, random_seed)
    metadata = from_files(features, targets, image_spec, halfwidth, n_train)
    write_metadata(directory, metadata)
    return 0


@cli.command()
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=1000)
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--halfwidth", type=int, default=1)
@click.argument("strip", type=int)
@click.argument("totalstrips", type=int)
def querydata(features: str, batchsize: int, nworkers: int,
              halfwidth: int, strip: int, totalstrips: int) -> int:
    """Grab a chunk for prediction."""
    pool = Pool(nworkers) if nworkers > 1 else DummyPool()
    log.info("Using {} worker processes".format(nworkers))

    dirname = os.path.basename(features).rsplit(".")[0] + \
        "_query{}of{}".format(strip, totalstrips)
    directory = os.path.join(os.getcwd(), dirname)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    image_spec = read_image_spec(features)
    tag = "query.{}of{}".format(strip, totalstrips)
    write_querydata(features, image_spec, strip, totalstrips,
                    batchsize, halfwidth, pool, directory, tag)
    pool.terminate()
    return 0
