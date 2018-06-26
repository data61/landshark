"""Dump data to hdf5 for external tools."""
import logging
from multiprocessing import cpu_count
import os

import click
import numpy as np
from typing import NamedTuple, Tuple

from landshark.scripts.logger import configure_logging
from landshark.dump import dump_training, dump_query
from landshark.featurewrite import read_featureset_metadata, \
    read_target_metadata
from landshark.metadata import TrainingMetadata, QueryMetadata, \
    CategoricalMetadata
from landshark.image import strip_image_spec
from landshark.util import mb_to_points
from landshark import errors
from landshark.trainingdata import CategoricalH5ArraySource, \
    OrdinalH5ArraySource, SourceMetadata
from landshark.kfold import KFolds


log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    batchMB: int
    nworkers: int


@click.group()
@click.option("--batch-mb", type=int, default=100)
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.pass_context
def cli(ctx: click.Context, verbosity: str, batch_mb: int,
        nworkers: int) -> None:
    """Parse the command line arguments."""
    ctx.obj = CliArgs(batchMB=batch_mb, nworkers=nworkers)
    configure_logging(verbosity)


@cli.command()
@click.option("--targets", type=click.Path(exists=True), required=True)
@click.option("--random_seed", type=int, default=666)
@click.option("--name", type=str, required=True)
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--nfolds", type=int)
@click.option("--halfwidth", type=int, default=0)
@click.pass_context
def trainingdata(ctx: click.Context, features: str, targets: str,
                 halfwidth: int, random_seed: int,
                 nfolds: int, name: str) -> None:
    """Dump training data."""
    catching_f = errors.catch_and_exit(trainingdata_entrypoint)
    catching_f(features, targets, halfwidth, random_seed, ctx.obj.nworkers,
               ctx.obj.batchMB, nfolds, name)


def trainingdata_entrypoint(features: str, targets: str, halfwidth: int,
                            random_seed: int, nworkers: int,
                            batchMB: int, nfolds: int, name: str) -> None:
    """Get training data."""
    feature_metadata = read_featureset_metadata(features)
    target_metadata = read_target_metadata(targets)

    ndim_ord = feature_metadata.ordinal.D \
        if feature_metadata.ordinal else 0
    ndim_cat = feature_metadata.categorical.D \
        if feature_metadata.categorical else 0
    active_feats_ord = np.ones(ndim_ord, dtype=bool)
    active_feats_cat = np.ones(ndim_cat, dtype=bool)
    points_per_batch = mb_to_points(batchMB, ndim_ord, ndim_cat,
                                    halfwidth=halfwidth)

    target_src = CategoricalH5ArraySource(targets) \
        if isinstance(target_metadata, CategoricalMetadata) \
        else OrdinalH5ArraySource(targets)

    n_rows = len(target_src)
    kfolds = KFolds(n_rows, nfolds, random_seed)
    tinfo = SourceMetadata(name, features, target_src,
                           feature_metadata.image, halfwidth, kfolds,
                           active_feats_ord, active_feats_cat)

    outfile_name = os.path.join(os.getcwd(),
                                "dump_traintest_" + name + ".hdf5")
    training_metadata = TrainingMetadata(targets=target_metadata,
                                         features=feature_metadata,
                                         halfwidth=halfwidth,
                                         nfolds=nfolds,
                                         testfold=1,
                                         fold_counts=tinfo.folds.counts)
    dump_training(tinfo, training_metadata,
                  outfile_name, points_per_batch, nworkers)
    log.info("Train/test dump complete")


@cli.command()
@click.option("--strip", type=int, nargs=2, default=(1, 1))
@click.option("--name", type=str, required=True)
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--halfwidth", type=int, default=0)
@click.pass_context
def querydata(ctx: click.Context, features: str, halfwidth: int,
              strip: Tuple[int, int], name: str) -> None:
    """Grab a chunk for prediction."""
    catching_f = errors.catch_and_exit(querydata_entrypoint)
    catching_f(features, halfwidth, strip, name, ctx.obj.batchMB,
               ctx.obj.nworkers)


def querydata_entrypoint(features: str, halfwidth: int,
                         strips: Tuple[int, int], name: str, batchMB: int,
                         nworkers: int) -> None:
    """Entry point for querydata dumping."""
    thisstrip, totalstrips = strips
    feature_metadata = read_featureset_metadata(features)

    ndim_ord = feature_metadata.ordinal.D \
        if feature_metadata.ordinal else 0
    ndim_cat = feature_metadata.categorical.D \
        if feature_metadata.categorical else 0
    points_per_batch = mb_to_points(batchMB, ndim_ord, ndim_cat,
                                    halfwidth=halfwidth)

    strip_imspec = strip_image_spec(thisstrip, totalstrips,
                                    feature_metadata.image)
    feature_metadata.image = strip_imspec

    fname = os.path.join(os.getcwd(), "dump_query_" + name +
                         "_strip{}of{}.hdf5".format(thisstrip, totalstrips))
    query_metadata = QueryMetadata(feature_metadata)
    dump_query(features, query_metadata, thisstrip, totalstrips,
               points_per_batch, halfwidth, nworkers, name, fname)
    log.info("Query dump complete")
