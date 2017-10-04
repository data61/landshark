"""Main landshark commands."""

import logging

import click

log = logging.getLogger(__name__)


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


@cli.command()
def learn() -> int:
    """Learn a model."""
    # TODO
    return 0


@cli.command()
def predict() -> int:
    """Predict using a learned model."""
    # TODO
    return 0
