"""Custom logging for landshark CLIs."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


def configure_logging(verbosity: str) -> None:
    """Configure the logger for STDOUT."""
    log = logging.getLogger("")
    tflog = logging.getLogger("tensorflow")
    tflog.handlers = []
    log.setLevel(verbosity)
    tflog.setLevel("ERROR")
    ch = logging.StreamHandler()
    formatter = ElapsedFormatter()
    ch.setFormatter(formatter)
    log.addHandler(ch)


class ElapsedFormatter(logging.Formatter):
    """Format logging message to include elapsed time."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Format incoming message."""
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated / 1000.0))
        msg = record.getMessage()
        logstr = "+{}s {}:{} {}".format(t, name, lvl, msg)
        return logstr
