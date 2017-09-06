"""Main landshark commands."""

import logging
import os.path

import click
# mypy type checking
from typing import List

from landshark import geoio

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
    logging.basicConfig()
    lg = logging.getLogger("")
    lg.setLevel(verbosity)
    return 0


def _tifnames(names: List[str]) -> List[str]:
    result = list(filter(lambda x: x.rsplit(".")[1] == "tif", names))
    return result


@cli.command()
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option("--name", type=str, required=True,
              help="Name of output file")
def build(files: List[str], name: str) -> int:
    """Build a tif stack from a set of input files."""
    out_filename = os.path.join(os.getcwd(), name + ".hdf5")
    tif_filenames = _tifnames(files)

    stack = geoio.ImageStack(tif_filenames)
    geoio.write_datafile(stack, out_filename)
    return 0


@cli.command()
@click.argument("fname", type=click.Path(exists=True))
def targets(fname: str) -> int:
    """Build a target file from shapefile."""
    out_filename = os.path.join(
        os.getcwd(), os.path.basename(fname).rsplit(".")[0] + ".hdf5")
    sf = geoio.ShapefileTargets(fname)
    geoio.write_targetfile(sf, out_filename)
    return 0
