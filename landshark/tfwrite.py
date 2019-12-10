"""Export data to tensorflow formats."""

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
import os.path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

FILESIZE_MB = 100


def query(data: Iterator[List[bytes]],
          n_total: int,
          output_directory: str,
          tag: str
          ) -> None:
    writer = _MultiFileWriter(output_directory, tag=tag)
    for d in data:
        writer.add(d)
    writer.close()


def training(data: Iterator[List[bytes]],
             n_total: int,
             output_directory: str,
             testfold: int,
             folds: Iterator[np.ndarray]
             ) -> None:
    test_directory = os.path.join(output_directory, "testing")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    writer = _MultiFileWriter(output_directory, tag="train")
    test_writer = _MultiFileWriter(test_directory, tag="test")

    for d, f in zip(data, folds):
        train_batch, test_batch = _split_on_mask(d, f, testfold)
        writer.add(train_batch)
        test_writer.add(test_batch)
    writer.close()
    test_writer.close()


def _get_mb(path: str) -> int:
    filesize = os.path.getsize(path) // (1024 ** 2)
    return filesize


class _MultiFileWriter:
    def __init__(self, output_directory: str, tag: str) -> None:
        self.output_directory = output_directory
        self.tag = tag
        self.file_index = -1
        self._options = tf.io.TFRecordOptions(
            tf.compat.v1.python_io.TFRecordCompressionType.ZLIB)
        self._f: Optional[tf.io.TFRecordWriter] = None
        self._nextfile()
        self.lines_written = 0

    def _nextfile(self) -> None:
        if self._f:
            self._f.close()
        self.file_index += 1
        self.path = os.path.join(
            self.output_directory,
            "{}.{:05d}.tfrecord".format(self.tag, self.file_index))
        self._f = tf.io.TFRecordWriter(self.path, options=self._options)

    def add(self, batch: List[bytes]) -> None:
        if self._f:
            filesize = _get_mb(self.path)
            if filesize > FILESIZE_MB:
                self._nextfile()
            for b in batch:
                self._f.write(b)
                self.lines_written += 1
            self._f.flush()
        else:
            raise RuntimeError("Cannot add data to writer that isnt open")

    def close(self) -> None:
        if self._f:
            self._f.close()
        else:
            raise RuntimeError("Cannot close a writer that isnt open")


def _split_on_mask(data: List[bytes],
                   folds: np.ndarray,
                   testfold: int
                   ) -> Tuple[List[bytes], List[bytes]]:
    mask = folds != testfold

    # in the case of one fold/no testing data
    if np.amax(folds) == 1:
        return data, data

    nmask = ~mask
    train_batch = [data[i] for i, m in enumerate(mask) if m]
    test_batch = [data[i] for i, m in enumerate(nmask) if m]
    return train_batch, test_batch
