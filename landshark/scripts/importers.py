"""Landshark importing commands."""

import logging
import os.path
from glob import glob
from multiprocessing import cpu_count
from copy import deepcopy

import tables
import click
from typing import List, Optional, Tuple, Set
import numpy as np

from landshark.tifread import shared_image_spec, OrdinalStackSource, \
    CategoricalStackSource
from landshark.featurewrite import write_ordinal, \
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
    write_ordinal_metadata, write_categorical_metadata, \
    read_featureset_metadata, read_target_metadata
from landshark.normalise import get_stats
from landshark.category import get_maps
from landshark.trainingdata import setup_training
from landshark.image import strip_image_spec

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
@click.option("--normalise/--no-normalise", is_flag=True, default=True)
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--ignore-crs/--no-ignore-crs", is_flag=True, default=False)
def tifs(categorical: str, ordinal: str, normalise: bool,
         name: str, nworkers: int, batchsize: int, ignore_crs: bool) -> int:
    """Build a tif stack from a set of input files."""
    log.info("Using {} worker processes".format(nworkers))
    out_filename = os.path.join(os.getcwd(), name + "_features.hdf5")
    ord_filenames = _tifnames(ordinal) if ordinal else []
    cat_filenames = _tifnames(categorical) if categorical else []
    all_filenames = ord_filenames + cat_filenames
    spec = shared_image_spec(all_filenames, ignore_crs)

    cat_meta, ord_meta = None, None
    N = None
    with tables.open_file(out_filename, mode="w", title=name) as outfile:
        if ordinal:
            ord_source = OrdinalStackSource(spec, ord_filenames)
            N = ord_source.shape[0] * ord_source.shape[1]
            log.info("Ordinal missing value is {}".format(ord_source.missing))
            if normalise:
                mean, var = get_stats(ord_source, batchsize)
                zvar = var == 0.0
                if any(zvar):
                    zsrcs = [c for z, c in zip(zvar, ord_source.columns) if z]
                    msg = 'The following sources have zero variance: {}'
                    raise ValueError(msg.format(zsrcs))
            else:
                stats = None

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
            if N is None:
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
@click.option("--categorical/--ordinal", is_flag=True, required=True)
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


def get_active_features(feature_metadata: FeatureSetMetadata,
                        withfeat: List[str], withoutfeat: List[str],
                        withlist: str) -> Tuple[np.ndarray, np.ndarray]:
    if len(withfeat) > 0 and len(withoutfeat) > 0:
        raise ValueError("Cant specificy withfeat and withoutfeat "
                         "at the same time")
    if withlist is not None and (len(withfeat) > 0 or len(withoutfeat) > 0):
        raise ValueError("Can't specify a feature list and command line "
                         "feature additions or subtractions")

    all_features: Set[str] = set()
    ncats = 0
    nords = 0
    if feature_metadata.ordinal is not None:
        all_features = all_features.union(set(feature_metadata.ordinal.labels))
        nords = len(feature_metadata.ordinal.labels)
    if feature_metadata.categorical is not None:
        all_features = all_features.union(
            set(feature_metadata.categorical.labels))
        ncats = len(feature_metadata.categorical.labels)
    if withlist is not None:
        feature_list = parse_withlist(withlist)
    elif len(withfeat) > 0:
        feature_list = withfeat
    elif len(withoutfeat) > 0:
        feature_list = list(all_features.difference(set(withoutfeat)))
    else:
        feature_list = list(all_features)
    feature_set = set(feature_list)
    if not feature_set.issubset(all_features):
        print("Error, the following requested features do not appear "
              " in the data:\n{}\n Possible features are:\n{}".format(
                  set(feature_list).difference(all_features), all_features))
        raise ValueError("Requested features not in data")

    ord_array = np.zeros(nords, dtype=bool)
    cat_array = np.zeros(ncats, dtype=bool)
    for f in feature_set:
        if feature_metadata.ordinal is not None:
            try:
                idx = feature_metadata.ordinal.labels.index(f)
                ord_array[idx] = 1
            except ValueError:
                pass
        if feature_metadata.categorical is not None:
            try:
                idx = feature_metadata.categorical.labels.index(f)
                cat_array[idx] = 1
            except ValueError:
                pass

    log.info("Selecting {} of {} ordinal features".format(
        np.sum(ord_array), nords))
    log.info("Selecting {} of {} categorical features".format(
        np.sum(cat_array), ncats))
    return ord_array, cat_array


def parse_withlist(listfile: str) -> List[str]:
    with open(listfile, "r") as f:
        lines = f.readlines()
    # remove the comment lines
    nocomments = [l.split("#")[0] for l in lines]
    stripped = [l.strip().rstrip() for l in nocomments]
    noempty = [l for l in stripped if l is not ""]
    return noempty


def _subset_ord_meta(m: OrdinalMetadata, active_ords: np.ndarray) \
        -> OrdinalMetadata:
    N = m.N
    D = np.sum(active_ords.astype(int))
    labels = [l for l, f in zip(m.labels, active_ords) if f]
    missing = m.missing
    means = m.means[active_ords] if m.means is not None else None
    variances = m.variances[active_ords] if m.variances is not None else None
    new_m = OrdinalMetadata(N, D, labels, missing, means, variances)
    return new_m


def _subset_cat_meta(m: CategoricalMetadata, active_cats: np.ndarray) \
        -> CategoricalMetadata:
    N = m.N
    D = np.sum(active_cats.astype(int))
    labels = [l for l, f in zip(m.labels, active_cats) if f]
    missing = m.missing
    ncategories = m.ncategories[active_cats]
    mappings = [e for e, f in zip(m.mappings, active_cats) if f]
    counts = [e for e, f in zip(m.counts, active_cats) if f]
    new_m = CategoricalMetadata(N, D, labels, missing, ncategories,
                                mappings, counts)
    return new_m


def active_column_metadata(m: FeatureSetMetadata, active_ords: np.ndarray,
                           active_cats: np.ndarray) -> FeatureSetMetadata:
    new_ordinal: Optional[OrdinalMetadata] = None
    new_categorical: Optional[CategoricalMetadata] = None
    if m.ordinal is not None and len(active_ords) > 0:
        new_ordinal = _subset_ord_meta(m.ordinal, active_ords)
    if m.categorical is not None and len(active_cats) > 0:
        new_categorical = _subset_cat_meta(m.categorical, active_cats)

    new_m = FeatureSetMetadata(new_ordinal, new_categorical, m.image)
    return new_m


@cli.command()
@click.argument("features", type=click.Path(exists=True))
@click.argument("targets", type=click.Path(exists=True))
@click.option("--folds", type=click.IntRange(2, None), default=10)
@click.option("--testfold", type=click.IntRange(1, None), default=1)
@click.option("--halfwidth", type=click.IntRange(0, None), default=1)
@click.option("--nworkers", type=click.IntRange(0, None), default=cpu_count())
@click.option("--batchsize", type=click.IntRange(1, None), default=100)
@click.option("--random_seed", type=int, default=666)
@click.option("--withfeat", type=str, multiple=True)
@click.option("--withoutfeat", type=str, multiple=True)
@click.option("--withlist", type=click.Path(exists=True))
@click.option("--name", type=str, required=True)
def trainingdata(features: str, targets: str, testfold: int,
                 folds: int, halfwidth: int, batchsize: int, nworkers: int,
                 random_seed: int, withfeat: List[str],
                 withoutfeat: List[str], withlist: str, name: str) -> int:
    """Get training data."""
    feature_metadata = read_featureset_metadata(features)
    target_metadata = read_target_metadata(targets)

    active_feats_ord, active_feats_cat = get_active_features(
        feature_metadata, withfeat, withoutfeat, withlist)

    reduced_feature_metadata = active_column_metadata(feature_metadata,
                                                      active_feats_ord,
                                                      active_feats_cat)

    tinfo = setup_training(features, feature_metadata,
                           targets, target_metadata, folds,
                           random_seed, halfwidth, active_feats_ord,
                           active_feats_cat)
    # TODO check this is being used correctly in the tensorflow regulariser
    n_train = len(tinfo.target_src) - tinfo.folds.counts[testfold]
    directory = os.path.join(os.getcwd(), name +
                             "_traintest{}of{}".format(testfold, folds))
    write_trainingdata(tinfo, directory, testfold, batchsize, nworkers)
    training_metadata = TrainingMetadata(targets=target_metadata,
                                         features=reduced_feature_metadata,
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
@click.option("--strip", type=int, nargs=2, default=(1, 1))
@click.option("--withfeat", type=str, multiple=True)
@click.option("--withoutfeat", type=str, multiple=True)
@click.option("--withlist", type=click.Path(exists=True))
@click.option("--name", type=str, required=True)
def querydata(features: str, batchsize: int, nworkers: int,
              halfwidth: int, strip: Tuple[int, int],
              withfeat: List[str], withoutfeat: List[str],
              withlist: str, name: str) -> int:
    strip_idx, totalstrips = strip
    assert strip_idx > 0 and strip_idx <= totalstrips

    """Grab a chunk for prediction."""
    log.info("Using {} worker processes".format(nworkers))

    dirname = name + "_query{}of{}".format(strip_idx, totalstrips)
    directory = os.path.join(os.getcwd(), dirname)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    feature_metadata = read_featureset_metadata(features)
    active_ord, active_cat = get_active_features(feature_metadata, withfeat,
                                                 withoutfeat, withlist)
    reduced_metadata = active_column_metadata(feature_metadata,
                                              active_ord,
                                              active_cat)

    strip_imspec = strip_image_spec(strip_idx, totalstrips,
                                    feature_metadata.image)
    reduced_metadata.image = strip_imspec
    tag = "query.{}of{}".format(strip_idx, totalstrips)
    write_querydata(features, feature_metadata.image, strip_idx, totalstrips,
                    batchsize, halfwidth, nworkers, directory, tag, active_ord,
                    active_cat)
    # TODO other info here like strips and windows
    query_metadata = QueryMetadata(reduced_metadata)
    pickle_metadata(directory, query_metadata)
    log.info("Query import complete")
    return 0
