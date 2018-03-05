"""Landshark importing commands."""

import logging
import os.path
from glob import glob
from multiprocessing import cpu_count

import tables
import click
# mypy type checking
from typing import List

# from landshark.basetypes import ClassSpec
from landshark.tifread import shared_image_spec, OrdinalStackSource, \
    CategoricalStackSource

# from landshark.hread import ImageFeatures
from landshark.featurewrite import write_imagespec, write_ordinal, \
    write_categorical, write_coordinates
from landshark.shpread import OrdinalShpArraySource, \
    CategoricalShpArraySource, CoordinateShpArraySource
# from landshark.importers import tfwrite
# from landshark.importers import metadata as mt
from landshark.scripts.logger import configure_logging
# from landshark import feed
from landshark.hread import read_image_spec, OrdinalH5ArraySource, CategoricalH5ArraySource
from landshark.trainingdata import write_trainingdata, write_querydata
from landshark.metadata import from_files, write_metadata
# from landshark.image import strip_image_spec

from landshark.normalise import get_stats
from landshark.category import get_maps

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
@click.option("--batchsize", type=int, default=100)
@click.option("--categorical", type=click.Path(exists=True))
@click.option("--ordinal", type=click.Path(exists=True))
@click.option("--nonormalise", is_flag=True)
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--nworkers", type=int, default=cpu_count())
def tifs(categorical: str, ordinal: str, nonormalise: bool,
         name: str, nworkers: int, batchsize: int) -> int:
    """Build a tif stack from a set of input files."""
    normalise = not nonormalise
    log.info("Using {} worker processes".format(nworkers))
    out_filename = os.path.join(os.getcwd(), name + "_features.hdf5")
    otmp_filename = os.path.join(os.getcwd(), name + "_features_ORAW.hdf5")
    ctmp_filename = os.path.join(os.getcwd(), name + "_features_CRAW.hdf5")
    ord_filenames = _tifnames(ordinal) if ordinal else []
    cat_filenames = _tifnames(categorical) if categorical else []
    all_filenames = ord_filenames + cat_filenames
    spec = shared_image_spec(all_filenames)

    with tables.open_file(out_filename, mode="w", title=name) as outfile:
        write_imagespec(spec, outfile)

    if ordinal and not normalise:
        log.info("Writing unnormalised ordinal data to output file")
        with tables.open_file(out_filename, mode="r+", title=name) as outfile:
            ord_source = OrdinalStackSource(spec, ord_filenames)
            write_ordinal(ord_source, outfile, nworkers, batchsize)

    elif ordinal and normalise:
        log.info("Writing unnormalised ordinal data to temporary file")
        with tables.open_file(otmp_filename, mode="w", title=name) as tmpfile:
            ord_source = OrdinalStackSource(spec, ord_filenames)
            write_ordinal(ord_source, tmpfile, nworkers, batchsize)
        # Compute stats with temp file
        tmp_src = OrdinalH5ArraySource(otmp_filename)
        stats = get_stats(tmp_src, batchsize, nworkers)
        with tables.open_file(out_filename, mode="r+") as outfile:
            log.info("Writing normalised ordinal data to output file")
            write_ordinal(tmp_src, outfile, nworkers, batchsize, stats)
        # Delete temp file!
        os.remove(otmp_filename)

    if categorical:
        with tables.open_file(ctmp_filename, mode="w", title=name) as tmpfile:
            log.info("Writing unmapped categorical data to temporary file")
            cat_source = CategoricalStackSource(spec, cat_filenames)
            write_categorical(cat_source, tmpfile, nworkers, batchsize)
        # Compute mapping with temp file
        tmp_src = CategoricalH5ArraySource(ctmp_filename)
        maps = get_maps(tmp_src, batchsize, nworkers)
        with tables.open_file(out_filename, mode="r+") as outfile:
            log.info("Writing mapped categorical data to output file")
            write_categorical(tmp_src, outfile, nworkers, batchsize, maps)
        # Delete temp file!
        os.remove(ctmp_filename)

    log.info("GTiff import complete")

    return 0


@cli.command()
@click.argument("targets", type=str, nargs=-1)
@click.option("--shapefile", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=100)
@click.option("--name", type=str, required=True)
@click.option("--every", type=int, default=1)
@click.option("--categorical", is_flag=True)
@click.option("--normalise", is_flag=True)
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--random_seed", type=int, default=666)
def targets(shapefile: str, batchsize: int, targets: List[str], name: str,
            every: int, categorical: bool, nworkers: int, normalise: bool,
            random_seed: int) -> int:
    """Build target file from shapefile."""
    log.info("Loading shapefile targets")
    log.info("Using {} worker processes".format(nworkers))
    out_filename = os.path.join(os.getcwd(), name + "_targets.hdf5")
    tmp_filename = os.path.join(os.getcwd(), name + "_targets_RAW.hdf5")

    with tables.open_file(out_filename, mode="w", title=name) as h5file:
        coord_src = CoordinateShpArraySource(shapefile, random_seed)
        write_coordinates(coord_src, h5file, batchsize)

        if not categorical and not normalise:
            log.info("Writing unnormalised ordinal targets to output file")
            ord_source = OrdinalShpArraySource(
                shapefile, targets, random_seed)
            write_ordinal(ord_source, h5file, batchsize)

    if categorical:
        with tables.open_file(tmp_filename, mode="w", title=name) as tmpfile:
            cat_source = CategoricalShpArraySource(
                shapefile, targets, random_seed)
            write_categorical(cat_source, tmpfile, batchsize)
        tmp_src = CategoricalH5ArraySource(tmp_filename)
        maps = get_maps(tmp_src, batchsize, nworkers)
        with tables.open_file(out_filename, mode="r+") as outfile:
            log.info("Writing mapped categorical targets to output file")
            write_categorical(tmp_src, outfile, nworkers, batchsize, maps)
        # Delete temp file!
        os.remove(tmp_filename)

    elif normalise:
        with tables.open_file(tmp_filename, mode="w", title=name) as tmpfile:
            ord_source = OrdinalShpArraySource(
                shapefile, targets, random_seed)
            write_ordinal(ord_source, tmpfile, batchsize)
        tmp_src = OrdinalH5ArraySource(tmp_filename)
        stats = get_stats(tmp_src, batchsize, nworkers)
        with tables.open_file(out_filename, mode="r+") as outfile:
            log.info("Writing normalised ordinal targets to output file")
            write_ordinal(tmp_src, outfile, nworkers, batchsize, stats)
        # Delete temp file!
        os.remove(tmp_filename)

    log.info("Target import complete")

    return 0


@cli.command()
@click.argument("features", type=click.Path(exists=True))
@click.argument("targets", type=click.Path(exists=True))
@click.option("--folds", type=click.IntRange(2, None), default=10)
@click.option("--testfold", type=click.IntRange(1, None), default=1)
@click.option("--halfwidth", type=click.IntRange(0, None), default=1)
@click.option("--nworkers", type=click.IntRange(0, None), default=cpu_count())
@click.option("--batchsize", type=click.IntRange(1, None), default=100)
@click.option("--random_seed", type=int, default=666)
def trainingdata(features: str, targets: str, testfold: int,
                 folds: int, halfwidth: int, batchsize: int, nworkers: int,
                 random_seed: int) -> int:
    """Get training data."""
    log.info("Using {} worker processes".format(nworkers))
    name = os.path.basename(features).rsplit("_features.")[0] + "-" + \
        os.path.basename(targets).rsplit("_targets.")[0]
    directory = os.path.join(os.getcwd(), name +
                             "_traintest{}of{}".format(testfold, folds))

    image_spec = read_image_spec(features)
    n_train = write_trainingdata(features, targets, image_spec, batchsize,
                                 halfwidth, nworkers, directory,
                                 testfold, folds, random_seed)
    metadata = from_files(features, targets, image_spec, halfwidth, n_train)
    write_metadata(directory, metadata)
    log.info("Training import complete")
    return 0


@cli.command()
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=1)
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--halfwidth", type=int, default=1)
@click.argument("strip", type=int)
@click.argument("totalstrips", type=int)
def querydata(features: str, batchsize: int, nworkers: int,
              halfwidth: int, strip: int, totalstrips: int) -> int:
    """Grab a chunk for prediction."""
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
                    batchsize, halfwidth, nworkers, directory, tag)
    log.info("Query import complete")
    return 0
