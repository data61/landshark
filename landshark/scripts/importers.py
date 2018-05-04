"""Landshark importing commands."""

import logging
import os.path
from glob import glob
from multiprocessing import cpu_count

import tables
import click
from typing import List, Optional

from landshark.tifread import shared_image_spec, OrdinalStackSource, \
    CategoricalStackSource
from landshark.featurewrite import write_imagespec, write_ordinal, \
    write_categorical, write_coordinates
from landshark.shpread import OrdinalShpArraySource, \
    CategoricalShpArraySource, CoordinateShpArraySource
from landshark.scripts.logger import configure_logging
from landshark.hread import read_image_spec, \
    CategoricalH5ArraySource, OrdinalH5ArraySource
from landshark.trainingdata import write_trainingdata, write_querydata
from landshark.metadata import OrdinalMetadata, \
    CategoricalMetadata, FeatureSetMetadata, TrainingMetadata, \
    QueryMetadata, pickle_metadata
from landshark.featurewrite import write_feature_metadata, \
    write_ordinal_metadata, write_categorical_metadata, read_featureset_metadata, \
    read_target_metadata
from landshark.normalise import get_stats
from landshark.category import get_maps
from landshark.trainingdata import setup_training

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


def _tifnames(directories: Optional[str]) -> List[str]:
    names: List[str] = []
    if directories is None:
        return names
    for d in directories:
        file_types = ("tif", "gtif")
        for t in file_types:
            glob_pattern = os.path.join(d, "**", "*.{}".format(t))
            names.extend(glob(glob_pattern, recursive=True))
    return names


@cli.command()
@click.option("--batchsize", type=int, default=100)
@click.option("--categorical", type=click.Path(exists=True), multiple=True)
@click.option("--ordinal", type=click.Path(exists=True), multiple=True)
@click.option("--nonormalise", is_flag=True)
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--ignore-crs/--no-ignore-crs", is_flag=True, default=False)
def tifs(categorical: str, ordinal: str, nonormalise: bool,
         name: str, nworkers: int, batchsize: int, ignore_crs: bool) -> int:
    """Build a tif stack from a set of input files."""
    normalise = not nonormalise
    log.info("Using {} worker processes".format(nworkers))
    out_filename = os.path.join(os.getcwd(), name + "_features.hdf5")
    ord_filenames = _tifnames(ordinal) if ordinal else []
    cat_filenames = _tifnames(categorical) if categorical else []
    all_filenames = ord_filenames + cat_filenames
    spec = shared_image_spec(all_filenames, ignore_crs)
    cat_meta, ord_meta = None, None
    N =  None
    with tables.open_file(out_filename, mode="w", title=name) as outfile:
        write_imagespec(spec, outfile)

        if ordinal:
            ord_source = OrdinalStackSource(spec, ord_filenames)
            N = ord_source.shape[0] * ord_source.shape[1]
            log.info("Ordinal missing value is {}".format(ord_source.missing))
            mean, var = None, None
            if normalise:
                mean, var = get_stats(ord_source, batchsize)
                zvar = var == 0.0
                if any(zvar):
                    zsrcs = [c for z, c in zip(zvar, ord_source.columns) if z]
                    msg = 'The following sources have zero variance: {}'
                    raise ValueError(msg.format(zsrcs))

            log.info("Writing normalised ordinal data to output file")
            ord_meta = OrdinalMetadata(N=N,
                                            D=ord_source.shape[-1],
                                              labels=ord_source.columns,
                                              missing=ord_source.missing,
                                              means=mean,
                                              variances=var)
            write_ordinal(ord_source, outfile, nworkers, batchsize)

        if categorical:
            cat_source = CategoricalStackSource(spec, cat_filenames)
            if N is not None:
                N = cat_source.shape[0] * cat_source.shape[1]
            log.info("Categorical missing value is {}".format(
                cat_source.missing))
            catdata = get_maps(cat_source, batchsize)
            maps, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in maps])
            log.info("Writing mapped categorical data to output file")
            cat_meta = CategoricalMetadata(N=N,
                                           D=cat_source.shape[-1],
                                                  labels=cat_source.columns,
                                                  missing=cat_source.missing,
                                                  ncategories=ncats,
                                                  mappings=maps,
                                                  counts=counts)
            write_categorical(cat_source, outfile, nworkers, batchsize, maps)

        assert N is not None
        meta = FeatureSetMetadata(ordinal=ord_meta, categorical=cat_meta,
                                  image=spec)
        write_feature_metadata(meta, outfile)

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
@click.option("--random_seed", type=int, default=666)
def targets(shapefile: str, batchsize: int, targets: List[str], name: str,
            every: int, categorical: bool, normalise: bool, random_seed: int) \
        -> int:
    """Build target file from shapefile."""
    log.info("Loading shapefile targets")
    out_filename = os.path.join(os.getcwd(), name + "_targets.hdf5")
    nworkers = 0  # shapefile reading breaks with concurrency

    with tables.open_file(out_filename, mode="w", title=name) as h5file:
        coord_src = CoordinateShpArraySource(shapefile, random_seed)
        write_coordinates(coord_src, h5file, batchsize)

        if categorical:
            cat_source = CategoricalShpArraySource(
                shapefile, targets, random_seed)
            catdata = get_maps(cat_source, batchsize)
            mappings, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in mappings])
            write_categorical(cat_source, h5file, nworkers, batchsize,
                              mappings)
            cat_meta = CategoricalMetadata(N=cat_source.shape[0],
                                             D=cat_source.shape[-1],
                                             labels=cat_source.columns,
                                             ncategories=ncats,
                                             mappings=mappings,
                                             counts=counts,
                                             missing=None)
            write_categorical_metadata(cat_meta, h5file)
        else:
            ord_source = OrdinalShpArraySource(shapefile, targets, random_seed)
            mean, var = get_stats(ord_source, batchsize) \
                if normalise else None, None
            write_ordinal(ord_source, h5file, nworkers, batchsize)
            ord_meta = OrdinalMetadata(N=ord_source.shape[0],
                                         D=ord_source.shape[-1],
                                         labels=ord_source.columns,
                                         means=mean,
                                         variances=var,
                                         missing=None)
            write_ordinal_metadata(ord_meta, h5file)
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
    feature_metadata = read_featureset_metadata(features)
    target_metadata = read_target_metadata(targets)
    tinfo = setup_training(features, feature_metadata,
                           targets, target_metadata, folds,
                           random_seed, halfwidth)
    n_train = len(tinfo.target_src) - tinfo.folds.counts[testfold]
    directory = os.path.join(os.getcwd(), tinfo.name +
                             "_traintest{}of{}".format(testfold, folds))
    write_trainingdata(tinfo, directory, testfold, batchsize, nworkers)
    training_metadata = TrainingMetadata(targets=target_metadata,
                                         features=feature_metadata,
                                         halfwidth=halfwidth,
                                         nfolds=folds,
                                         testfold=testfold,
                                         fold_counts=tinfo.folds.counts)
    pickle_metadata(directory, training_metadata)
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

    feature_metadata = read_featureset_metadata(features)
    tag = "query.{}of{}".format(strip, totalstrips)
    write_querydata(features, feature_metadata.image, strip, totalstrips,
                    batchsize, halfwidth, nworkers, directory, tag)
    # TODO other info here like strips and windows
    query_metadata = QueryMetadata(feature_metadata)
    pickle_metadata(directory, query_metadata)
    log.info("Query import complete")
    return 0
