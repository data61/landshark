"""Dump data to hdf5 for external tools."""

import logging
import os
from multiprocessing import cpu_count
from typing import NamedTuple, Tuple

import click
import numpy as np

from landshark import errors
from landshark.dataprocess import SourceMetadata
from landshark.dump import dump_query, dump_training
from landshark.featurewrite import (read_featureset_metadata,
                                    read_target_metadata)
from landshark.hread import CategoricalH5ArraySource, OrdinalH5ArraySource
from landshark.image import strip_image_spec
from landshark.kfold import KFolds
from landshark.metadata import (CategoricalMetadata, QueryMetadata,
                                TrainingMetadata)
from landshark.scripts.logger import configure_logging
from landshark.util import mb_to_points

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    batchMB: float
    nworkers: int


@click.group()
@click.option("--nworkers", type=click.IntRange(0, None), default=cpu_count(),
              help="Number of additional worker processes")
@click.option("--batch-mb", type=float, default=100,
              help="Approximate size in megabytes of data read per "
              "worker per iteration")
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.pass_context
def cli(ctx: click.Context, verbosity: str, batch_mb: float,
        nworkers: int) -> None:
    """Dump (patched) training or query data into an export HDF5 file."""
    ctx.obj = CliArgs(batchMB=batch_mb, nworkers=nworkers)
    configure_logging(verbosity)


@cli.command()
@click.option("--targets", type=click.Path(exists=True), required=True,
              help="Target HDF5 file from which to read")
@click.option("--random_seed", type=int, default=666,
              help="Random state for assigning data to folds")
@click.option("--name", type=str, required=True,
              help="Name of the output folder")
@click.option("--features", type=click.Path(exists=True), required=True,
              help="Feature HDF5 file from which to read")
@click.option("--halfwidth", type=int, default=0,
              help="half width of patch size. Patch side length is "
              "2 x halfwidth + 1")
@click.option("--nfolds", type=int, default=10, help="The number of folds "
              "into which to assign each training point.")
@click.pass_context
def traintest(ctx: click.Context, features: str, targets: str,
              halfwidth: int, random_seed: int,
              nfolds: int, name: str) -> None:
    """Dump training data (including fold assignments) into an HDF5 file."""
    catching_f = errors.catch_and_exit(traintest_entrypoint)
    catching_f(features, targets, halfwidth, random_seed, ctx.obj.nworkers,
               ctx.obj.batchMB, nfolds, name)


def traintest_entrypoint(features: str, targets: str, halfwidth: int,
                         random_seed: int, nworkers: int,
                         batchMB: float, nfolds: int, name: str) -> None:
    """Get training data."""
    feature_metadata = read_featureset_metadata(features)
    target_metadata = read_target_metadata(targets)

    ndim_ord = feature_metadata.D_ordinal
    ndim_cat = feature_metadata.D_categorical
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
@click.option("--strip", type=int, nargs=2, default=(1, 1),
              help="Horizontal strip of the image, eg --strip 3 5 is the "
              "third strip of 5")
@click.option("--name", type=str, required=True,
              help="The name of the output from this command.")
@click.option("--features", type=click.Path(exists=True), required=True,
              help="Feature HDF5 file from which to read")
@click.option("--halfwidth", type=int, default=0,
              help="half width of patch size. Patch side length is "
              "2 x halfwidth + 1")
@click.pass_context
def query(ctx: click.Context, features: str, halfwidth: int,
          strip: Tuple[int, int], name: str) -> None:
    """Export query data to an HDF5 file."""
    catching_f = errors.catch_and_exit(query_entrypoint)
    catching_f(features, halfwidth, strip, name, ctx.obj.batchMB,
               ctx.obj.nworkers)


def query_entrypoint(features: str, halfwidth: int,
                     strips: Tuple[int, int], name: str, batchMB: float,
                     nworkers: int) -> None:
    """Entry point for querydata dumping."""
    thisstrip, totalstrips = strips
    feature_metadata = read_featureset_metadata(features)

    ndim_ord = feature_metadata.D_ordinal
    ndim_cat = feature_metadata.D_categorical
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


if __name__ == "__main__":
    cli()
