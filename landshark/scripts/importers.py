"""Landshark importing commands."""

import logging
import os.path

import click
# mypy type checking
from typing import List

from landshark.importers.tifread import ImageStack
from landshark.importers.featurewrite import write_datafile
from landshark.importers.shpread import train_test_targets
from landshark.importers.targetwrite import write_targetfile
from landshark.scripts.logger import configure_logging

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


def _tifnames(names: List[str]) -> List[str]:
    result = list(filter(lambda x: x.rsplit(".")[1] == "tif", names))
    return result


@cli.command()
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option("--name", type=str, required=True,
              help="Name of output file")
@click.option('--standardise/--no-standardise', default=True,
              help="Standardise the input features")
def tifs(files: List[str], name: str, standardise: bool) -> int:
    """Build a tif stack from a set of input files."""
    out_filename = os.path.join(os.getcwd(), name + ".hdf5")
    tif_filenames = _tifnames(files)

    stack = ImageStack(tif_filenames)
    write_datafile(stack, out_filename, standardise)
    return 0


@cli.command()
@click.argument("fname", type=click.Path(exists=True))
@click.option("--test_frac", type=float, default=0.1)
@click.option("--random_seed", type=int, default=666)
def shapefile(fname: str, test_frac: float, random_seed: int) -> int:
    """Build a target file from shapefile."""
    file_str = os.path.basename(fname).rsplit(".")[0]
    tr_filename = os.path.join(os.getcwd(), file_str + "_train" + ".hdf5")
    ts_filename = os.path.join(os.getcwd(), file_str + "_test" + ".hdf5")
    sf_tr, sf_ts = train_test_targets(fname, test_frac, random_seed)
    write_targetfile(sf_tr, tr_filename)
    write_targetfile(sf_ts, ts_filename)
    return 0
