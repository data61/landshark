import os
import click
import numpy as np
import logging
from landshark.scripts.logger import configure_logging
from multiprocessing import cpu_count
from landshark.featurewrite import read_featureset_metadata, \
    read_target_metadata

from landshark.trainingdata import write_trainingdata, write_querydata
from landshark.metadata import OrdinalMetadata, \
    CategoricalMetadata, FeatureSetMetadata, TrainingMetadata, \
    QueryMetadata, pickle_metadata
from landshark.trainingdata import setup_training
from landshark.image import strip_image_spec
from typing import List, Tuple, Set, Optional

log = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
def cli(verbosity: str) -> int:
    """Parse the command line arguments."""
    configure_logging(verbosity)
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
