"""Landshark importing commands."""

import logging
import os.path
from multiprocessing import cpu_count
from typing import List, NamedTuple, Tuple

import click
import numpy as np
import tables

from landshark import errors
from landshark import metadata as meta
from landshark.category import get_maps
from landshark.featurewrite import (write_categorical,
                                    write_coordinates,
                                    write_continuous,
                                    write_feature_metadata,
                                    write_target_metadata)
from landshark.fileio import tifnames
from landshark.normalise import get_stats
from landshark.scripts.logger import configure_logging
from landshark.shpread import (CategoricalShpArraySource,
                               CoordinateShpArraySource, ContinuousShpArraySource)
from landshark.tifread import (CategoricalStackSource, ContinuousStackSource,
                               shared_image_spec)
from landshark.util import mb_to_points, mb_to_rows

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
        nworkers: int, batch_mb: float) -> int:
    """Import features and targets into landshark-compatible formats."""
    log.info("Using a maximum of {} worker processes".format(nworkers))
    ctx.obj = CliArgs(nworkers, batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option("--categorical", type=click.Path(exists=True), multiple=True,
              help="Directory containing categorical geotifs")
@click.option("--continuous", type=click.Path(exists=True), multiple=True,
              help="Directory containing continuous geotifs")
@click.option("--normalise/--no-normalise", is_flag=True, default=True,
              help="Normalise the continuous tif bands")
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--ignore-crs/--no-ignore-crs", is_flag=True, default=False,
              help="Ignore CRS (projection and datum) information")
@click.pass_context
def tifs(ctx: click.Context, categorical: Tuple[str, ...],
         continuous: Tuple[str, ...], normalise: bool, name: str,
         ignore_crs: bool) -> None:
    """Build a tif stack from a set of input files."""
    nworkers = ctx.obj.nworkers
    batchMB = ctx.obj.batchMB
    cat_list = list(categorical)
    con_list = list(continuous)
    catching_f = errors.catch_and_exit(tifs_entrypoint)
    catching_f(nworkers, batchMB, cat_list,
               con_list, normalise, name, ignore_crs)


def tifs_entrypoint(nworkers: int, batchMB: float, categorical: List[str],
                    continuous: List[str], normalise: bool, name: str,
                    ignore_crs: bool) -> None:
    """Entrypoint for tifs without click cruft."""
    out_filename = os.path.join(os.getcwd(), "features_{}.hdf5".format(name))

    con_filenames = tifnames(list(continuous))
    cat_filenames = tifnames(list(categorical))
    log.info("Found {} continuous TIF files".format(len(con_filenames)))
    log.info("Found {} categorical TIF files".format(len(cat_filenames)))
    has_con = len(con_filenames) > 0
    has_cat = len(cat_filenames) > 0
    all_filenames = con_filenames + cat_filenames
    if not len(all_filenames) > 0:
        raise errors.NoTifFilesFound()

    N_con, N_cat = None, None
    con_meta, cat_meta = None, None
    spec = shared_image_spec(all_filenames, ignore_crs)

    with tables.open_file(out_filename, mode="w", title=name) as outfile:
        if has_con:
            con_source = ContinuousStackSource(spec, con_filenames)
            ndims_con = con_source.shape[-1]
            con_rows_per_batch = mb_to_rows(batchMB, spec.width, ndims_con, 0)
            N_con = con_source.shape[0] * con_source.shape[1]
            N = N_con
            log.info("Continuous missing value set to {}".format(
                con_source.missing))
            stats = None
            if normalise:
                stats = get_stats(con_source, con_rows_per_batch)
                mean, var = stats
                if any(var == 0.0):
                    raise errors.ZeroVariance(var, con_source.columns)
                log.info("Writing normalised continuous data to output file")
            else:
                log.info("Writing unnormalised continuous data to output file")
            con_meta = meta.ContinuousFeatureSet(labels=con_source.columns,
                                          missing=con_source.missing,
                                          means=mean,
                                          variances=var)
            write_continuous(con_source, outfile, nworkers, con_rows_per_batch,
                          stats)

        if has_cat:
            cat_source = CategoricalStackSource(spec, cat_filenames)
            N_cat = cat_source.shape[0] * cat_source.shape[1]
            N = N_cat
            if N_con and N_cat != N_con:
                raise errors.ConCatNMismatch(N_con, N_cat)

            ndims_cat = cat_source.shape[-1]
            cat_rows_per_batch = mb_to_rows(batchMB, spec.width, 0, ndims_cat)
            log.info("Categorical missing value set to {}".format(
                cat_source.missing))
            catdata = get_maps(cat_source, cat_rows_per_batch)
            maps, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in maps])
            log.info("Writing mapped categorical data to output file")
            cat_meta = meta.CategoricalFeatureSet(labels=cat_source.columns,
                                           missing=cat_source.missing,
                                           nvalues=ncats,
                                           mappings=maps,
                                           counts=counts)
            write_categorical(cat_source, outfile, nworkers,
                              cat_rows_per_batch, maps)
        m = meta.FeatureSet(continuous=con_meta, categorical=cat_meta,
                            image=spec, N=N, halfwidth=0)
        write_feature_metadata(m, outfile)
    log.info("Tif import complete")


@cli.command()
@click.option("--record", type=str, multiple=True, required=True,
              help="Label of record to extract as a target")
@click.option("--shapefile", type=click.Path(exists=True), required=True,
              help="Path to .shp file for reading")
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option("--every", type=int, default=1, help="Subsample (randomly)"
              " by this factor, e.g. every 2 samples half the points")
@click.option("--dtype", type=click.Choice(["continuous", "categorical"]),
              required=True, help="The type of the targets")
@click.option("--normalise", is_flag=True, help="Normalise each target."
              " Only relevant for continuous targets.")
@click.option("--random_seed", type=int, default=666, help="The random seed "
              "for shuffling targets on import")
@click.pass_context
def targets(ctx: click.Context, shapefile: str, record: Tuple[str, ...],
            name: str, every: int, dtype: str, normalise: bool,
            random_seed: int) -> None:
    """Build target file from shapefile."""
    record_list = list(record)
    categorical = dtype == "categorical"
    batchMB = ctx.obj.batchMB
    catching_f = errors.catch_and_exit(targets_entrypoint)
    catching_f(batchMB, shapefile, record_list, name, every, categorical,
               normalise, random_seed)


def targets_entrypoint(batchMB: float, shapefile: str, records: List[str],
                       name: str, every: int, categorical: bool,
                       normalise: bool, random_seed: int) -> None:
    """Targets entrypoint without click cruft."""
    log.info("Loading shapefile targets")
    out_filename = os.path.join(os.getcwd(), "targets_{}.hdf5".format(name))
    nworkers = 0  # shapefile reading breaks with concurrency

    with tables.open_file(out_filename, mode="w", title=name) as h5file:
        log.info("Reading shapefile point coordinates")
        cocon_src = CoordinateShpArraySource(shapefile, random_seed)
        cocon_batchsize = mb_to_points(batchMB, ndim_con=0,
                                       ndim_cat=0, ndim_coord=2)
        write_coordinates(cocon_src, h5file, cocon_batchsize)

        if categorical:
            log.info("Reading shapefile categorical records")
            cat_source = CategoricalShpArraySource(
                shapefile, records, random_seed)
            cat_batchsize = mb_to_points(batchMB, ndim_con=0,
                                         ndim_cat=cat_source.shape[-1])
            catdata = get_maps(cat_source, cat_batchsize)
            mappings, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in mappings])
            write_categorical(cat_source, h5file, nworkers, cat_batchsize,
                              mappings)
            cat_meta = meta.CategoricalTarget(N=cat_source.shape[0],
                                              labels=cat_source.columns,
                                              nvalues=ncats,
                                              mappings=mappings,
                                              counts=counts)
            write_target_metadata(cat_meta, h5file)
        else:
            log.info("Reading shapefile continuous records")
            con_source = ContinuousShpArraySource(shapefile, records, random_seed)
            con_batchsize = mb_to_points(batchMB,
                                         ndim_con=con_source.shape[-1],
                                         ndim_cat=0)
            mean, var = get_stats(con_source, con_batchsize) \
                if normalise else None, None
            write_continuous(con_source, h5file, nworkers, con_batchsize)
            con_meta = meta.ContinuousTarget(N=con_source.shape[0],
                                             labels=con_source.columns,
                                             means=mean,
                                             variances=var)
            write_target_metadata(con_meta, h5file)
    log.info("Target import complete")


if __name__ == "__main__":
    cli()
