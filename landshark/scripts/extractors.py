"""Import tifs and targets into landshark world."""

import logging
import os
from multiprocessing import cpu_count
from typing import List, NamedTuple, Optional, Tuple

import click

from landshark import errors
from landshark.dataprocess import SourceMetadata
from landshark.datawrite import write_querydata, write_trainingdata
from landshark.extract import active_column_metadata, get_active_features
from landshark.featurewrite import (read_featureset_metadata,
                                    read_target_metadata)
from landshark.hread import CategoricalH5ArraySource, ContinuousH5ArraySource
from landshark.image import strip_image_spec
from landshark.kfold import KFolds
from landshark.metadata import (CategoricalMetadata, QueryMetadata,
                                TrainingMetadata, pickle_metadata)
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
@click.option("--withfeat", type=str, multiple=True,
              help="Whitelist a particular feature. "
              "Can be given multiple times.")
@click.option("--withoutfeat", type=str, multiple=True,
              help="Blacklist a particular feature. "
              "Can be given multiple times")
@click.option("--withlist", type=click.Path(exists=True), help="Path to text"
              " file giving newline separated list of features to use")
@click.pass_context
def traintest(ctx: click.Context, targets: str, split: Tuple[int, ...],
              random_seed: int, name: str, features: str,
              withfeat: Tuple[str, ...], withoutfeat: Tuple[str, ...],
              withlist: Optional[str], halfwidth: int) -> int:
    """Extract training and testing data to train and validate a model."""
    fold, nfolds = split
    catching_f = errors.catch_and_exit(traintest_entrypoint)
    catching_f(targets, fold, nfolds, random_seed, list(withfeat),
               list(withoutfeat), withlist, name, halfwidth,
               ctx.obj.nworkers, features, ctx.obj.batchMB)


def traintest_entrypoint(targets: str, testfold: int, folds: int,
                         random_seed: int, withfeat: List[str],
                         withoutfeat: List[str], withlist: Optional[str],
                         name: str, halfwidth: int, nworkers: int,
                         features: str, batchMB: float) -> None:
    """Get training data."""
    feature_metadata = read_featureset_metadata(features)
    target_metadata = read_target_metadata(targets)

    active_feats_con, active_feats_cat = get_active_features(
        feature_metadata, withfeat, withoutfeat, withlist)

    reduced_feature_metadata = active_column_metadata(feature_metadata,
                                                      active_feats_con,
                                                      active_feats_cat)
    ndim_con = reduced_feature_metadata.D_continuous
    ndim_cat = reduced_feature_metadata.D_categorical
    points_per_batch = mb_to_points(batchMB, ndim_con, ndim_cat,
                                    halfwidth=halfwidth)

    target_src = CategoricalH5ArraySource(targets) \
        if isinstance(target_metadata, CategoricalMetadata) \
        else ContinuousH5ArraySource(targets)

    n_rows = len(target_src)
    kfolds = KFolds(n_rows, folds, random_seed)
    tinfo = SourceMetadata(name, features, target_src,
                           feature_metadata.image, halfwidth, kfolds,
                           active_feats_con, active_feats_cat)
    # TODO check this is being used correctly in the tensorflow regulariser
    n_train = len(tinfo.target_src) - tinfo.folds.counts[testfold]
    directory = os.path.join(os.getcwd(), "traintest_{}_fold{}of{}".format(
        name, testfold, folds))
    write_trainingdata(tinfo, directory, testfold, points_per_batch, nworkers)
    training_metadata = TrainingMetadata(targets=target_metadata,
                                         features=reduced_feature_metadata,
                                         halfwidth=halfwidth,
                                         nfolds=folds,
                                         testfold=testfold,
                                         fold_counts=tinfo.folds.counts)
    pickle_metadata(directory, training_metadata)
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
@click.option("--withfeat", type=str, multiple=True,
              help="Whitelist a particular feature. "
              "Can be given multiple times.")
@click.option("--withoutfeat", type=str, multiple=True,
              help="Blacklist a particular feature. "
              "Can be given multiple times")
@click.option("--withlist", type=click.Path(exists=True), help="Path to text"
              " file giving newline separated list of features to use")
@click.pass_context
def query(ctx: click.Context, strip: Tuple[int, int], name: str,
          features: str, halfwidth: int, withfeat: Tuple[str, ...],
          withoutfeat: Tuple[str, ...], withlist: Optional[str]) -> None:
    """Extract query data for making prediction images."""
    catching_f = errors.catch_and_exit(query_entrypoint)
    catching_f(features, ctx.obj.batchMB, ctx.obj.nworkers,
               halfwidth, strip, list(withfeat), list(withoutfeat),
               withlist, name)


def query_entrypoint(features: str, batchMB: float, nworkers: int,
                     halfwidth: int, strip: Tuple[int, int],
                     withfeat: List[str], withoutfeat: List[str],
                     withlist: str, name: str) -> int:
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

    feature_metadata = read_featureset_metadata(features)
    active_con, active_cat = get_active_features(feature_metadata, withfeat,
                                                 withoutfeat, withlist)

    reduced_metadata = active_column_metadata(feature_metadata,
                                              active_con,
                                              active_cat)
    ndim_con = reduced_metadata.D_continuous
    ndim_cat = reduced_metadata.D_categorical
    points_per_batch = mb_to_points(batchMB, ndim_con, ndim_cat,
                                    halfwidth=halfwidth)

    strip_imspec = strip_image_spec(strip_idx, totalstrips,
                                    feature_metadata.image)
    reduced_metadata.image = strip_imspec
    tag = "query.{}of{}".format(strip_idx, totalstrips)

    qinfo = SourceMetadata(name, features, None, strip_imspec,
                           halfwidth, None, active_con, active_cat)

    write_querydata(qinfo, directory, strip_idx, totalstrips,
                    points_per_batch, nworkers, tag)

    # TODO other info here like strips and windows
    query_metadata = QueryMetadata(reduced_metadata)
    pickle_metadata(directory, query_metadata)
    log.info("Query import complete")
    return 0


if __name__ == "__main__":
    cli()
