"""Write training and query data to tensorflow records."""

import logging

import numpy as np

from landshark import tfwrite
from landshark.basetypes import IdReader
from landshark.dataprocess import (SerialisingQueryDataProcessor,
                                   SerialisingTrainingDataProcessor,
                                   SourceMetadata)
from landshark.image import ImageSpec, indices_strip
from landshark.iteration import batch_slices
from landshark.multiproc import task_list

log = logging.getLogger(__name__)


def write_trainingdata(tinfo: SourceMetadata,
                       output_directory: str,
                       testfold: int,
                       batchsize: int,
                       nworkers: int
                       ) -> None:

    log.info("Testing data is fold {} of {}".format(testfold, tinfo.folds))
    log.info("Writing training data to tfrecord in {}-point batches".format(
        batchsize))
    n_rows = len(tinfo.target_src)
    worker = SerialisingTrainingDataProcessor(tinfo)
    tasks = list(batch_slices(batchsize, n_rows))
    out_it = task_list(tasks, tinfo.target_src, worker, nworkers)
    fold_it = tinfo.folds.iterator(batchsize)
    tfwrite.training(out_it, n_rows, output_directory, testfold, fold_it)


def write_querydata(feature_path: str,
                    image_spec: ImageSpec,
                    strip: int,
                    total_strips: int,
                    points_per_batch: int,
                    halfwidth: int,
                    n_workers: int,
                    output_directory: str,
                    tag: str,
                    active_con: np.ndarray,
                    active_cat: np.ndarray) -> None:
    reader_src = IdReader()
    it, n_total = indices_strip(image_spec, strip, total_strips,
                                points_per_batch)
    worker = SerialisingQueryDataProcessor(image_spec, feature_path, halfwidth,
                                           active_con, active_cat)
    tasks = list(it)
    out_it = task_list(tasks, reader_src, worker, n_workers)
    tfwrite.query(out_it, n_total, output_directory, tag)
