"""Import tifs and targets into landshark world."""

import os
import click
import logging
from typing import NamedTuple, List, Optional
from landshark.scripts.logger import configure_logging
from multiprocessing import cpu_count
from landshark.featurewrite import read_featureset_metadata, \
    read_target_metadata

from landshark.trainingdata import write_trainingdata, write_querydata
from landshark.metadata import TrainingMetadata, \
    QueryMetadata, pickle_metadata, CategoricalMetadata
from landshark.image import strip_image_spec
from typing import List, Tuple, Optional
from landshark import errors
from landshark.extract import get_active_features, active_column_metadata
from landshark.trainingdata import CategoricalH5ArraySource, \
    OrdinalH5ArraySource, SourceMetadata
from landshark.kfold import KFolds

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    nworkers: int
    batchsize: int
    features: str
    halfwidth: int
    withfeat: List[str]
    withoutfeat: List[str]
    withlist: Optional[str]


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.option("--features", type=click.Path(exists=True), required=True)
@click.option("--batchsize", type=int, default=1)
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--halfwidth", type=int, default=1)
@click.option("--withfeat", type=str, multiple=True)
@click.option("--withoutfeat", type=str, multiple=True)
@click.option("--withlist", type=click.Path(exists=True))
@click.pass_context
def cli(ctx: click.Context, verbosity: str, features: str,
        batchsize: int, nworkers: int, halfwidth: int,
        withfeat: Tuple[str, ...], withoutfeat: Tuple[str, ...],
        withlist: Optional[str]) -> int:
    """Parse the command line arguments."""
    ctx.obj = CliArgs(nworkers, batchsize, features, halfwidth, list(withfeat),
                      list(withoutfeat), withlist)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--targets", type=click.Path(exists=True), required=True)
@click.option("--split", type=int, nargs=2)
@click.option("--random_seed", type=int, default=666)
@click.option("--name", type=str, required=True)
@click.pass_context
def traintest(ctx: click.Context, targets: str, split: Tuple[int, ...],
              random_seed: int, name: str) -> int:
    fold, nfolds = split
    catching_f = errors.catch_and_exit(traintest_entrypoint)
    catching_f(targets, fold, nfolds, random_seed, ctx.obj.withfeat,
               ctx.obj.withoutfeat, ctx.obj.withlist, name, ctx.obj.halfwidth,
               ctx.obj.nworkers, ctx.obj.features, ctx.obj.batchsize)


def traintest_entrypoint(targets: str, testfold: int, folds: int,
                         random_seed: int, withfeat: List[str],
                         withoutfeat: List[str], withlist: Optional[str],
                         name: str, halfwidth: int, nworkers: int,
                         features: str, batchsize: int) -> None:
    """Get training data."""
    feature_metadata = read_featureset_metadata(features)
    target_metadata = read_target_metadata(targets)

    active_feats_ord, active_feats_cat = get_active_features(
        feature_metadata, withfeat, withoutfeat, withlist)

    reduced_feature_metadata = active_column_metadata(feature_metadata,
                                                      active_feats_ord,
                                                      active_feats_cat)

    target_src = CategoricalH5ArraySource(targets) \
        if isinstance(target_metadata, CategoricalMetadata) \
        else OrdinalH5ArraySource(targets)

    n_rows = len(target_src)
    kfolds = KFolds(n_rows, folds, random_seed)
    tinfo = SourceMetadata(name, features, target_src,
                           feature_metadata.image, halfwidth, kfolds,
                           active_feats_ord, active_feats_cat)
    # TODO check this is being used correctly in the tensorflow regulariser
    n_train = len(tinfo.target_src) - tinfo.folds.counts[testfold]
    directory = os.path.join(os.getcwd(), "traintest_{}_fold{}of{}".format(
        name, testfold, folds))
    write_trainingdata(tinfo, directory, testfold, batchsize, nworkers)
    training_metadata = TrainingMetadata(targets=target_metadata,
                                         features=reduced_feature_metadata,
                                         halfwidth=halfwidth,
                                         nfolds=folds,
                                         testfold=testfold,
                                         fold_counts=tinfo.folds.counts)
    pickle_metadata(directory, training_metadata)
    log.info("Training import complete")


@cli.command()
@click.option("--strip", type=int, nargs=2, default=(1, 1))
@click.option("--name", type=str, required=True)
@click.pass_context
def query(ctx: click.Context, strip: Tuple[int, int], name: str) -> None:
    catching_f = errors.catch_and_exit(query_entrypoint)
    catching_f(ctx.obj.features, ctx.obj.batchsize, ctx.obj.nworkers,
               ctx.obj.halfwidth, strip, ctx.obj.withfeat, ctx.obj.withoutfeat,
               ctx.obj.withlist, name)


def query_entrypoint(features: str, batchsize: int, nworkers: int,
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
