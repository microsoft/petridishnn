# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import zmq
import sys, os
import time
import multiprocessing as mp
import atexit, weakref

from tensorpack.utils import logger
from tensorpack.dataflow.parallel import _bind_guard, _get_pipe_name, del_weakref, _zmq_catch_error
from tensorpack.utils.concurrency import (ensure_proc_terminate,
                                 mask_sigint, start_proc_mask_signal,
                                 StoppableThread)
from tensorpack.utils.serialize import loads, dumps
from tensorpack.dataflow.base import DataFlowReentrantGuard

__all__ = ['MultiSourcePool', 'WhileSleepWorker', 'PetridishServerIPC', 'stop_mark_fn',
    'has_stopped', 'mark_stopped', 'mark_failed', 'is_mark_failure',
    'TRAIN_HALLU', 'TRAIN_MODEL', 'TRAIN_CRITIC_MODEL',
    'TRAIN_CRITIC_HALLU', 'TRAIN_CRITIC_PARENT','NUM_POOLS']

TRAIN_HALLU = 0
TRAIN_MODEL = 1
TRAIN_CRITIC_MODEL = 2
TRAIN_CRITIC_HALLU = 3
TRAIN_CRITIC_PARENT = 4
NUM_POOLS = 5

class MultiSourcePool(object):
        """
        This is for managing multiple pools of resources. In petridish project, this is for
        managing proc of each task and jobs on local vs on remote.
        """

        def __init__(self, pool_sizes):
            """
            ns : the limit on each resource stored in a list.
            """
            self.queues = []
            self.limits = pool_sizes
            for _ in self.limits:
                self.queues.append(set())

        def has_idle(self, qid=None):
            if qid is None:
                return any(map(lambda x : len(x[0]) < x[1], zip(self.queues, self.limits)))
            elif isinstance(qid, list):
                return any(map(lambda i : len(self.queues[i]) < self.limits[i], qid))
            else:
                return len(self.queues[qid]) < self.limits[qid]

        def num_idle(self, qid):
            return self.limits[qid] - len(self.queues[qid])

        def has_active(self):
            return any(map(lambda x : len(x) > 0, self.queues))

        def enqueue(self, qid, val):
            self.queues[qid].add(val)

        def dequeue(self, qid, val):
            self.queues[qid].discard(val)


class WhileSleepWorker(mp.Process):
    def __init__(self, conn_name, hwm, entry_func, stop_func, msg_func, sleep_time):
        super(WhileSleepWorker, self).__init__()
        self.conn_name = conn_name
        self.hwm = hwm
        self.entry_func = entry_func
        self.stop_func = stop_func
        self.msg_func = msg_func
        self.sleep_time = sleep_time

    def run(self):
        self.entry_func()
        while not self.stop_func():
            time.sleep(self.sleep_time)
        msg = self.msg_func()
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.set_hwm(self.hwm)
        socket.connect(self.conn_name)
        try:
            socket.send(msg, copy=True)
        except KeyboardInterrupt:
            pass
        finally:
            # http://zguide.zeromq.org/page:all
            # The proper way is to set a low LINGER value (1 second), and then close the socket.
            #
            # Technically, socket will close itself on garbage collect.
            time.sleep(2)
            socket.close(1)
            context.destroy(1)


class PetridishServerIPC(object):
    """
    Since the GPU server farm does not allow a job to launch more jobs, and does not have
    in-memory or on-network communication between jobs, we have to use file system to
    communicate between processes.
    Hence, instead of using callback of remote jobs to communicate back to the main, we
    have the following communication structure:

    Main will spawn a sleeper worker for each remeote job.
    Remote job will be launched via a job crawler that locates on the local machine.
    Remote job will finish, and mark_stopped() in its log dir.
    The sleeper associated with the remote job on the main job will read the log dir
    and use msg_func() to generate message that is sent back to the main process.

    PetridishServerIPC handles the above operations as follows.

    Initialize : start the server communication so that sleeper can talk back to main.
    spawn : launches the sleepers.
    get_finished_message : listen to sleepers.
    _parse_msg : parse the returned msg into actual output, and parts that IPC cares.
    _join_finished : resource reaping on the finished sleepers.
    """

    def __init__(self, pool_sizes, hwm):
        """
        Args:
            ns (list of int) : the limit of each proc pool.
            hwm (int) : the high water mark for zmq.
        """
        self.pools = MultiSourcePool(pool_sizes)
        self.hwm = hwm
        self._initialized = False
        self._guard = DataFlowReentrantGuard()
        self.worker_id_to_proc = dict()
        atexit.register(del_weakref, weakref.ref(self))


    def initialize(self):
        """
        """
        if self._initialized:
            return
        self._initialized = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self.hwm)
        self.pipename = _get_pipe_name('petridish_mainloop_{}'.format(time.time()))
        self.worker_id = 0
        _bind_guard(self.socket, self.pipename)


    def spawn(self, job_type, entry_func, stop_func, msg_func, sleep_time):
        qid = job_type
        self.worker_id += 1
        msg_func2 = lambda : dumps(msg_func() + [ self.worker_id ])
        proc = WhileSleepWorker(self.pipename, self.hwm,
            entry_func, stop_func, msg_func2, sleep_time)
        self.pools.enqueue(qid, self.worker_id)
        self.worker_id_to_proc[self.worker_id] = proc
        start_proc_mask_signal([proc])


    def get_finished_message(self):
        with self._guard, _zmq_catch_error('PetridishServerIPC'):
            msg = loads(self.socket.recv(copy=True))
        output, job_type, worker_id = self._parse_msg(msg)
        self._join_finished(job_type, worker_id)
        return output, job_type


    def _parse_msg(self,msg):
        """
        return output, jobtype, job_id
        """
        return msg[0:-2], msg[-2], msg[-1]


    def _join_finished(self, job_type, worker_id):
        proc = self.worker_id_to_proc.pop(worker_id, None)
        self.pools.dequeue(job_type, worker_id)
        if proc is None:
            raise Exception('msg is from a finished job????')
        proc.join()


    def __del__(self):
        try:
            if not self._initialized:
                return
            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)
            for wid in self.worker_id_to_proc:
                x = self.worker_id_to_proc[wid]
                x.terminate()
                x.join(5)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception:
            pass



def stop_mark_fn(dirname, is_interrupted=False):
    if is_interrupted:
        return os.path.join(dirname, 'interrupted.bin')
    return os.path.join(dirname, 'finished.bin')


def has_stopped(log_dir, is_interrupted=False):
    fn = stop_mark_fn(log_dir, is_interrupted)
    return os.path.exists(fn)


def mark_stopped(log_dir, is_interrupted=False, msg_func=None):
    fn = stop_mark_fn(log_dir, is_interrupted)
    tmp_fn = fn + '.tmp'
    with open(tmp_fn, 'wb') as fout:
        msg = msg_func() if msg_func is not None else dumps('meow')
        fout.write(msg)
    # we do this in case we cannot finish writing "finish.bin" before it is found
    os.rename(tmp_fn, fn)


def mark_failed(log_dir):
    fn = stop_mark_fn(log_dir, is_interrupted=False)
    tmp_fn = fn + '.tmp'
    with open(tmp_fn, 'wb') as fout:
        msg = dumps('failed_meow')
        fout.write(msg)
    # we do this in case we cannot finish writing "finish.bin" before it is found
    os.rename(tmp_fn, fn)


def is_mark_failure(log_dir):
    """
    Return:

    is_failed (bool) : whether the marke_stopped file indicates a failure
    ss (bytes) or ret (obj) : if the file content is json-loads capable, then we return the
        loaded json/obj, ret = loads(fin.read()); else we return the bytes in the file
        ss = fin.read()
    """
    fn = stop_mark_fn(log_dir, is_interrupted=False)
    if not os.path.exists(fn):
        return True, None
    with open(fn, 'rb') as fin:
        ss = fin.read()
        try:
            ret = loads(ss)
        except:
            return False, ss
        return ret == 'failed_meow', ret
