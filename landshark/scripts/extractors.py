"""Import tifs and targets into landshark world."""

import logging
import os
from multiprocessing import cpu_count
from typing import NamedTuple, Tuple

import click

from landshark import errors
from landshark import metadata as meta
from landshark.dataprocess import (ProcessQueryArgs, ProcessTrainingArgs,
                                   write_querydata, write_trainingdata)
from landshark.featurewrite import read_feature_metadata, read_target_metadata
from landshark.hread import CategoricalH5ArraySource, ContinuousH5ArraySource
from landshark.image import strip_image_spec
from landshark.kfold import KFolds
from landshark.scripts.logger import configure_logging
from landshark.util import mb_to_points

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    nworkers: int
    batchMB: float


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.option("--nworkers", type=click.IntRange(0, None), default=cpu_count(),
              help="Number of additional worker processes")
@click.option("--batch-mb", type=float, default=10,
              help="Approximate size in megabytes of data read per "
              "worker per iteration")
@click.pass_context
def cli(ctx: click.Context, verbosity: str,
        batch_mb: float, nworkers: int) -> int:
    """Extract features and targets for training, testing and prediction."""
    ctx.obj = CliArgs(nworkers, batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--targets", type=click.Path(exists=True), required=True,
              help="Target HDF5 file from which to read")
@click.option("--split", type=int, nargs=2, default=(1, 10),
              help="Train/test split fold structure. Firt argument is test "
              "fold (counting from 1), second is total folds.")
@click.option("--random_seed", type=int, default=666,
              help="Random state for assigning data to folds")
@click.option("--name", type=str, required=True,
              help="Name of the output folder")
@click.option("--features", type=click.Path(exists=True), required=True,
              help="Feature HDF5 file from which to read")
@click.option("--halfwidth", type=int, default=0,
              help="half width of patch size. Patch side length is "
              "2 x halfwidth + 1")
@click.pass_context
def traintest(ctx: click.Context, targets: str, split: Tuple[int, ...],
              random_seed: int, name: str,
              features: str, halfwidth: int) -> None:
    """Extract training and testing data to train and validate a model."""
    fold, nfolds = split
    catching_f = errors.catch_and_exit(traintest_entrypoint)
    catching_f(targets, fold, nfolds, random_seed,name, halfwidth,
               ctx.obj.nworkers, features, ctx.obj.batchMB)


def traintest_entrypoint(targets: str, testfold: int, folds: int,
                         random_seed: int, name: str, halfwidth: int,
                         nworkers: int, features: str, batchMB: float) -> None:
    """Get training data."""
    feature_metadata = read_feature_metadata(features)
    feature_metadata.halfwidth = halfwidth
    target_metadata = read_target_metadata(targets)

    ndim_con = len(feature_metadata.continuous.columns) \
        if feature_metadata.continuous else 0
    ndim_cat = len(feature_metadata.categorical.columns) \
        if feature_metadata.categorical else 0
    points_per_batch = mb_to_points(batchMB, ndim_con, ndim_cat,
                                    halfwidth=halfwidth)

    target_src = CategoricalH5ArraySource(targets) \
        if isinstance(target_metadata, meta.CategoricalTarget) \
        else ContinuousH5ArraySource(targets)

    n_rows = len(target_src)
    kfolds = KFolds(n_rows, folds, random_seed)

    n_train = len(target_src) - kfolds.counts[testfold]
    directory = os.path.join(os.getcwd(), "traintest_{}_fold{}of{}".format(
        name, testfold, folds))

    args = ProcessTrainingArgs(name=name,
                               feature_path=features,
                               target_src=target_src,
                               image_spec=feature_metadata.image,
                               halfwidth=halfwidth,
                               testfold=testfold,
                               folds=kfolds,
                               directory=directory,
                               batchsize=points_per_batch,
                               nworkers=nworkers)
    write_trainingdata(args)
    training_metadata = meta.Training(targets=target_metadata,
                                      features=feature_metadata,
                                      nfolds=folds,
                                      testfold=testfold,
                                      fold_counts=kfolds.counts)
    training_metadata.save(directory)
    log.info("Training import complete")


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
def query(ctx: click.Context, strip: Tuple[int, int], name: str,
          features: str, halfwidth: int) -> None:
    """Extract query data for making prediction images."""
    catching_f = errors.catch_and_exit(query_entrypoint)
    catching_f(features, ctx.obj.batchMB, ctx.obj.nworkers,
               halfwidth, strip, name)


def query_entrypoint(features: str, batchMB: float, nworkers: int,
                     halfwidth: int, strip: Tuple[int, int],
                     name: str) -> int:
    strip_idx, totalstrips = strip
    assert strip_idx > 0 and strip_idx <= totalstrips

    """Grab a chunk for prediction."""
    log.info("Using {} worker processes".format(nworkers))

    dirname = "query_{}_strip{}of{}".format(name, strip_idx, totalstrips)
    directory = os.path.join(os.getcwd(), dirname)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    feature_metadata = read_feature_metadata(features)
    feature_metadata.halfwidth = halfwidth
    ndim_con = len(feature_metadata.continuous.columns) \
        if feature_metadata.continuous else 0
    ndim_cat = len(feature_metadata.categorical.columns) \
        if feature_metadata.categorical else 0
    points_per_batch = mb_to_points(batchMB, ndim_con, ndim_cat,
                                    halfwidth=halfwidth)

    strip_imspec = strip_image_spec(strip_idx, totalstrips,
                                    feature_metadata.image)
    feature_metadata.image = strip_imspec
    tag = "query.{}of{}".format(strip_idx, totalstrips)

    qargs = ProcessQueryArgs(name, features, strip_imspec,
                             strip_idx, totalstrips, halfwidth, directory,
                             points_per_batch, nworkers, tag)

    write_querydata(qargs)
    feature_metadata.save(directory)
    log.info("Query import complete")
    return 0


if __name__ == "__main__":
    cli()
