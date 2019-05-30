from collections import namedtuple
import heapq
import random
import copy
import numpy as np
import bisect

from petridish.nas_control.queue_diversity import diverse_top_k

__all__ = [
    'PetridishQueueEntry', 'PetridishQueue', 'PetridishHeapQueue',
    'PetridishSortedQueue', 'IDX_CNT', 'IDX_PQE', 'IDX_PV'
]

PetridishQueueEntry = namedtuple('PetridishQueueEntry',
    ['model_str', 'model_iter', 'parent_iter', 'search_depth'])

# priority value
IDX_PV = 0
# priority queue entry
IDX_PQE = 1
# counts of usage
IDX_CNT = 2

class PetridishQueue(object):
    """
    A queue that store the model layer info list in str format.
    """
    def __init__(self, name):
        self.name = name
        self.entries = []

    def add(self, model_str, model_iter, parent_iter, search_depth,
            priority, counts=0):
        pqe = PetridishQueueEntry(model_str=model_str, model_iter=model_iter,
            parent_iter=parent_iter, search_depth=search_depth)
        if np.isinf(priority):
            return # do not insert infinity
        if np.isnan(priority):
            priority = float('inf') # model overflow/underflow. So low priority
        item = [None] * 3
        item[IDX_PV] = priority
        item[IDX_PQE] = pqe
        item[IDX_CNT] = counts
        self._add(item)

    def _add(self, item):
        raise NotImplementedError()

    def size(self):
        return len(self.entries)

    def update(self, key=None, l_priority=None, full_sort=False, keep_top_k=None):
        """
        Sort the petridish queue using func

        Args
            func : a function that maps from (idx, petridish_queue_entry) to a float; the default
            is lambda i, _ : i, which means we follow the FIFO order.
        """
        if len(self.entries) == 0:
            return
        assert bool(key) != bool(l_priority), "only one option should be used for updating priority"
        if key:
            for i in range(self.size()):
                self.entries[i][IDX_PV] = key(self.entries[i][IDX_PQE])
        else:
            for i in range(self.size()):
                self.entries[i][IDX_PV] = l_priority[i]
        if full_sort:
            self.entries.sort()
            if keep_top_k is not None:
                self.entries[keep_top_k:] = []
        elif keep_top_k is not None:
            self.entries = heapq.nsmallest(keep_top_k, self.entries)
        else:
            self._update()

    def _update(self):
        raise NotImplementedError()

    def pop(self):
        ret = self._pop()
        return ret[IDX_PQE]

    def _pop(self):
        raise NotImplementedError()

    def peek(self):
        return self.peek_at(0)

    def peek_at(self, loc):
        self.entries[loc][IDX_CNT] += 1
        return self.entries[loc][IDX_PQE]

    def random_peek(self):
        loc = random.randrange(0, self.size())
        return self.peek_at(loc)

    def all(self, full_info=False):
        if full_info:
            return [x for x in self.entries]
        return [x[IDX_PQE] for x in self.entries]

    def all_as_generator(self, full_info=False):
        if full_info:
            for x in self.entries:
                yield x
        else:
            for x in self.entries:
                yield x[IDX_PQE]

    def diverse_top_k(self, mi_info, div_opts):
        l_mi = [ent[IDX_PQE].model_iter for ent in self.entries]
        l_priority = [ent[IDX_PV] for ent in self.entries]
        top_indices = diverse_top_k(l_mi, l_priority, mi_info, div_opts)
        return [copy.deepcopy(self.entries[idx][IDX_PQE]) for idx in top_indices]

    def keep_top_k(self, k):
        self.entries = heapq.nsmallest(k, self.entries)

    def remove_slice(self, to_rm):
        # a hack way to force remove slice
        if self.size() == 0:
            return

        if to_rm.stop > to_rm.start:
            l_priority = [ent[IDX_PV] for ent in self.entries]
            n_rm = to_rm.stop - to_rm.start
            l_priority[to_rm] = [float('inf') for _ in range(n_rm)]
            self.update(
                l_priority=l_priority, full_sort=True,
                keep_top_k=self.size() - n_rm)

    def keep_indices_no_auto_update(self, indices):
        self.entries = [self.entries[i] for i in indices]
        # It's up to caller to call update to maintain the ordering.


class PetridishHeapQueue(PetridishQueue):

    def __init__(self, name):
        super(PetridishHeapQueue, self).__init__(name)

    def _update(self):
        heapq.heapify(self.entries)

    def _add(self, item):
        heapq.heappush(self.entries, item)

    def _pop(self):
        return heapq.heappop(self.entries)


class PetridishSortedQueue(PetridishQueue):

    def __init__(self, name):
        super(PetridishSortedQueue, self).__init__(name)

    def _update(self):
        self.entries.sort()

    def _add(self, item):
        idx = bisect.bisect_left(self.entries, item)
        self.entries[idx:idx] = [item]

    def _pop(self):
        ret = self.entries[0]
        self.entries[0:1] = []
        return ret

    def keep_top_k(self, k):
        if self.size() <= k:
            return
        self.entries[k:] = []

    def remove_slice(self, to_rm):
        self.entries[to_rm] = []