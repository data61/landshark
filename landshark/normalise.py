import numpy as np
import logging

# from landshark.basetypes import ClassSpec
from landshark.multiproc import task_list
from landshark import iteration
from landshark.util import to_masked
from landshark.hread import OrdinalH5ArraySource

log = logging.getLogger(__name__)

class MeanPreprocessor:

    def __init__(self, missing_value):
        self.missing_value = missing_value

    def __call__(self, x):
        bs = x.reshape((-1, x.shape[-1]))
        bm = to_masked(bs, self.missing_value)
        count = np.ma.count(bm, axis=0)
        psum = np.ma.sum(bm, axis=0)
        return count, psum

class VarPreprocessor:

    def __init__(self, missing_value, mean):
        self.missing_value = missing_value
        self.mean = mean

    def __call__(self, x):
        bs = x.reshape((-1, x.shape[-1]))
        bm = to_masked(bs, self.missing_value)
        delta = bm - self.mean[np.newaxis, :]
        dsum = np.ma.sum(np.ma.power(delta, 2), axis=0)
        return dsum


class Normaliser:

    def __init__(self, mean, var):
        self._mean = mean
        self._var = var

    def __call__(self, x):
        x0 = x - self._mean
        xw = x0 / self._var
        return xw


def get_stats(src, batchsize, n_workers):
    log.info("Computing feature means")
    n_rows = src.shape[0]
    it = list(iteration.batch_slices(batchsize, n_rows))
    worker = MeanPreprocessor(src.missing)
    out_it = task_list(it, src, worker, n_workers)
    count_list, psum_list = zip(*list(out_it))
    full_counts = np.sum(np.array(count_list), axis=0)
    full_sums = np.sum(np.array(psum_list), axis=0)
    log.info("Computing feature variances")
    it = list(iteration.batch_slices(batchsize, n_rows))
    means = full_counts / full_sums
    worker = VarPreprocessor(src.missing, means)
    ssums = list(task_list(it, src, worker, n_workers))
    vars = np.sum(np.array(ssums), axis=0) / full_counts
    return means, vars

