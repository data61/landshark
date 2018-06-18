"""Landshark importing commands."""

import logging
import os.path
from multiprocessing import cpu_count

import tables
import click
import numpy as np
from typing import List, NamedTuple, Tuple

from landshark.tifread import shared_image_spec, OrdinalStackSource, \
    CategoricalStackSource
from landshark.featurewrite import write_ordinal, \
    write_categorical, write_coordinates
from landshark.shpread import OrdinalShpArraySource, \
    CategoricalShpArraySource, CoordinateShpArraySource
from landshark.metadata import OrdinalMetadata, \
    CategoricalMetadata, FeatureSetMetadata
from landshark.featurewrite import write_feature_metadata, \
    write_ordinal_metadata, write_categorical_metadata
from landshark.scripts.logger import configure_logging
from landshark.normalise import get_stats
from landshark.category import get_maps
from landshark.fileio import tifnames
from landshark import errors

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    nworkers: int
    batchsize: int


@click.group()
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
@click.option("--nworkers", type=int, default=cpu_count())
@click.option("--batchsize", type=int, default=100)
@click.pass_context
def cli(ctx: click.Context, verbosity: str,
        nworkers: int, batchsize: int) -> int:
    """Parse the command line arguments."""
    log.info("Using a maximum of {} worker processes".format(nworkers))
    log.info("Using a batchsize of {} rows".format(batchsize))
    ctx.obj = CliArgs(nworkers, batchsize)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--categorical", type=click.Path(exists=True), multiple=True)
@click.option("--ordinal", type=click.Path(exists=True), multiple=True)
@click.option("--normalise/--no-normalise", is_flag=True, default=True)
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--ignore-crs/--no-ignore-crs", is_flag=True, default=False)
@click.pass_context
def tifs(ctx: click.Context, categorical: Tuple[str, ...],
         ordinal: Tuple[str, ...], normalise: bool, name: str,
         ignore_crs: bool) -> None:
    """Build a tif stack from a set of input files."""
    nworkers = ctx.obj.nworkers
    batchsize = ctx.obj.batchsize
    cat_list = list(categorical)
    ord_list = list(ordinal)
    catching_f = errors.catch_and_exit(tifs_entrypoint)
    catching_f(nworkers, batchsize, cat_list,
               ord_list, normalise, name, ignore_crs)


def tifs_entrypoint(nworkers: int, batchsize: int, categorical: List[str],
                    ordinal: List[str], normalise: bool, name: str,
                    ignore_crs: bool) -> None:
    """Entrypoint for tifs without click cruft."""
    out_filename = os.path.join(os.getcwd(), "features_{}.hdf5".format(name))

    ord_filenames = tifnames(list(ordinal))
    cat_filenames = tifnames(list(categorical))
    log.info("Found {} ordinal TIF files".format(len(ord_filenames)))
    log.info("Found {} categorical TIF files".format(len(cat_filenames)))
    has_ord = len(ord_filenames) > 0
    has_cat = len(cat_filenames) > 0
    all_filenames = ord_filenames + cat_filenames
    if not len(all_filenames) > 0:
        raise errors.NoTifFilesFound()

    N_ord, N_cat = None, None
    ord_meta, cat_meta = None, None
    spec = shared_image_spec(all_filenames, ignore_crs)

    with tables.open_file(out_filename, mode="w", title=name) as outfile:
        if has_ord:
            ord_source = OrdinalStackSource(spec, ord_filenames)
            N_ord = ord_source.shape[0] * ord_source.shape[1]
            log.info("Ordinal missing value set to {}".format(
                ord_source.missing))
            mean, var = None, None
            if normalise:
                mean, var = get_stats(ord_source, batchsize)
                if any(var == 0.0):
                    raise errors.ZeroVariance(var, ord_source.columns)
            log.info("Writing normalised ordinal data to output file")
            ord_meta = OrdinalMetadata(N=N_ord,
                                       D=ord_source.shape[-1],
                                       labels=ord_source.columns,
                                       missing=ord_source.missing,
                                       means=mean,
                                       variances=var)
            write_ordinal(ord_source, outfile, nworkers, batchsize)

        if has_cat:
            cat_source = CategoricalStackSource(spec, cat_filenames)
            N_cat = cat_source.shape[0] * cat_source.shape[1]
            if N_ord and N_cat != N_ord:
                raise errors.OrdCatNMismatch(N_ord, N_cat)

            log.info("Categorical missing value set to {}".format(
                cat_source.missing))
            catdata = get_maps(cat_source, batchsize)
            maps, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in maps])
            log.info("Writing mapped categorical data to output file")
            cat_meta = CategoricalMetadata(N=N_cat,
                                           D=cat_source.shape[-1],
                                           labels=cat_source.columns,
                                           missing=cat_source.missing,
                                           ncategories=ncats,
                                           mappings=maps,
                                           counts=counts)
            write_categorical(cat_source, outfile, nworkers, batchsize, maps)
        meta = FeatureSetMetadata(ordinal=ord_meta, categorical=cat_meta,
                                  image=spec)
        write_feature_metadata(meta, outfile)
    log.info("Tif import complete")


@cli.command()
@click.option("--record", type=str, multiple=True)
@click.option("--shapefile", type=click.Path(exists=True), required=True)
@click.option("--name", type=str, required=True)
@click.option("--every", type=int, default=1)
@click.option("--dtype", type=click.Choice(["ordinal", "categorical"]),
              required=True)
@click.option("--normalise", is_flag=True)
@click.option("--random_seed", type=int, default=666)
@click.pass_context
def targets(ctx: click.Context, shapefile: str, record: Tuple[str, ...],
            name: str, every: int, dtype: str, normalise: bool,
            random_seed: int) -> None:
    """Build target file from shapefile."""
    record_list = list(record)
    categorical = dtype == "categorical"
    batchsize = ctx.obj.batchsize
    catching_f = errors.catch_and_exit(targets_entrypoint)
    catching_f(batchsize, shapefile, record_list, name, every, categorical,
               normalise, random_seed)


def targets_entrypoint(batchsize: int, shapefile: str, records: List[str],
                       name: str, every: int, categorical: bool,
                       normalise: bool, random_seed: int) -> None:
    """Targets entrypoint without click cruft."""
    log.info("Loading shapefile targets")
    out_filename = os.path.join(os.getcwd(), "targets_{}.hdf5".format(name))
    nworkers = 0  # shapefile reading breaks with concurrency

    with tables.open_file(out_filename, mode="w", title=name) as h5file:
        coord_src = CoordinateShpArraySource(shapefile, random_seed)
        write_coordinates(coord_src, h5file, batchsize)

        if categorical:
            cat_source = CategoricalShpArraySource(
                shapefile, records, random_seed)
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
            ord_source = OrdinalShpArraySource(shapefile, records, random_seed)
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
