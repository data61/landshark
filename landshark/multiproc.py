"""Multiprocessing task queue implementation."""

import queue
import logging
from mulitprocessing import Pipe  # type: ignore
from multiprocessing import Process, Queue
from typing import List, Dict, Iterator, Any

from landshark.basetypes import Reader, Worker

from tqdm import tqdm

log = logging.getLogger(__name__)

# Do not make result queue size 0 if you care about memory
# Values larger than 1 probably dont help anyway
RESULT_QUEUE_SIZE = 1

# We're assuming the actual request objects are small here
REQ_QUEUE_SIZE = 0


class _Task(Process):

    def __init__(self, datasrc: Reader,
                 f: Worker, in_queue: Queue,
                 out_queue: Queue, shutdown: Pipe,
                 blocktime: float=0.1) -> None:
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shutdown = shutdown
        self.datasrc = datasrc
        self.f = f
        self._blocktime = blocktime
        super().__init__()

    def run(self) -> None:
        running = True
        with self.datasrc:
            while running:
                if self.shutdown.poll():
                    running = False

                try:
                    task_id, req = self.in_queue.get(True, self._blocktime)
                    data = self.datasrc(req)
                    out_data = self.f(data)
                    self.out_queue.put((task_id, out_data))
                except queue.Empty:
                    pass


def task_list(task_list: List[Any], reader: Reader, worker: Worker,
              n_workers: int) -> Iterator[Any]:
    if n_workers == 0:
        return _task_list_0(task_list, reader, worker)
    else:
        return _task_list_multi(task_list, reader, worker, n_workers)


def _task_list_0(task_list: List[Any], reader: Reader,
                 worker: Worker) -> Iterator[Any]:
    total = len(task_list)
    with reader:
        with tqdm(total=total) as pbar:
            for t in task_list:
                data = reader(t)
                output = worker(data)
                yield output
                pbar.update()


def _task_list_multi(task_list: List[Any], reader: Reader, worker: Worker,
                     n_workers: int) -> Iterator[Any]:
    req_queue = Queue(REQ_QUEUE_SIZE)
    result_queue = Queue(RESULT_QUEUE_SIZE)
    shutdown_recv, shutdown_send = Pipe(False)
    worker_procs = [_Task(reader, worker, req_queue, result_queue,
                          shutdown_recv)
                    for _ in range(n_workers)]
    cache: Dict[int, Any] = {}

    task_id = 0
    task_id_out = 0
    for w in worker_procs:
        w.start()

    total = len(task_list)
    for x in task_list:
        req_queue.put((task_id, x))
        task_id += 1

    with tqdm(total=total) as pbar:
        for _ in range(task_id):
            satisfied = False
            while not satisfied:
                if task_id_out in cache:
                    result = cache.pop(task_id_out)
                    satisfied = True
                    task_id_out += 1
                else:
                    task_id, result = result_queue.get()
                    cache[task_id] = result
            yield result
            pbar.update()

    shutdown_send.send(0)
    for w in worker_procs:
        w.join()
