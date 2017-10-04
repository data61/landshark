"""Custom logging for landshark CLIs."""

import logging


def configure_logging(verbosity: str) -> None:
    """Configure the logger for STDOUT."""
    log = logging.getLogger("")
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
