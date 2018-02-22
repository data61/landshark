"""Multiprocessing task queue implementation."""

import queue
import logging
from multiprocessing import Process, Queue, Pipe
from typing import NamedTuple, List, Dict, Any

from tqdm import tqdm

log = logging.getLogger(__name__)



class _Task(Process):

    def __init__(self, class_spec, in_queue, out_queue, shutdown,
                 blocktime=0.01):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shutdown = shutdown
        self.class_spec = class_spec
        super().__init__()
        self._blocktime = blocktime


    def run(self):

        f = self.class_spec.instantiate()
        running = True
        while running:
            if self.shutdown.poll():
                running = False

            try:
                task_id, in_data = self.in_queue.get(True, self._blocktime)
                out_data = f(in_data)
                self.out_queue.put((task_id, out_data))
            except queue.Empty:
                pass


def task_list(task_list, reader_spec, worker_spec, n_workers,
              req_queue_size=0, data_queue_size=1, result_queue_size=1):
    req_queue = Queue(req_queue_size)
    data_queue = Queue(data_queue_size)
    result_queue = Queue(result_queue_size)
    shutdown_recv, shutdown_send = Pipe(False)
    io_process = _Task(reader_spec, req_queue, data_queue, shutdown_recv)
    worker_procs = [_Task(worker_spec, data_queue, result_queue, shutdown_recv)
                    for _ in range(n_workers)]
    cache = {}

    task_id = 0
    task_id_out = 0
    io_process.start()
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
    io_process.join()
    for w in worker_procs:
        w.join()
