"""Custom logging for landshark CLIs."""

from os import environ
import logging

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def configure_logging(verbosity: str) -> None:
    """Configure the logger for STDOUT."""
    log = logging.getLogger("")
    tflog = logging.getLogger("tensorflow")
    tflog.handlers = []
    log.setLevel(verbosity)
    ch = logging.StreamHandler()
    formatter = ElapsedFormatter()
    ch.setFormatter(formatter)
    log.addHandler(ch)


class ElapsedFormatter(logging.Formatter):
    """Format logging message to include elapsed time."""

    def format(self, record: logging.LogRecord) -> str:
        """Format incoming message."""
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated / 1000.0))
        msg = record.getMessage()
        logstr = "+{}s {}:{} {}".format(t, name, lvl, msg)
        return logstr
