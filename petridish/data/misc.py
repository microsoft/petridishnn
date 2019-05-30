# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import scipy.io.arff as arff
import bisect
import os, sys
import json

from tensorpack.dataflow import RNGDataFlow, BatchData, PrefetchData, imgaug
from tensorpack.callbacks import Inferencer

def preprocess_data_flow(ds, options, is_train, do_multiprocess=False):
    ds_size = ds.size()
    while options.batch_size > ds_size:
        options.batch_size //= 2
    ds = BatchData(ds, max(1, options.batch_size // options.nr_gpu),
        remainder=not is_train)
    if do_multiprocess:
        ds = PrefetchData(ds, 5, 5)
    return ds

class Cutout(imgaug.ImageAugmentor):

    def __init__(self, length=8, n_holes=1):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
        if len(img.shape) == 3:
            mask = np.reshape(mask, [h, w, 1])
        img = img * mask
        return img


class AgeOnlyCallBack(Inferencer):

    def __init__(self, n_layers, save_fn=None):

        self.names = [ 'input', 'label', 'layer{:03d}.0.pred/preds:0'.format(n_layers - 1)]
        self.save_fn = save_fn

    def _before_inference(self):
        if self.save_fn:
            self.fout = open(self.save_fn, 'wt')
        else:
            self.fout = sys.stdout

    def _get_fetches(self):
        return self.names

    def _on_fetches(self, output):
        #self.fout.write("{}\t{}\t{}\n".format(output[0], output[1], output[2]))
        self.fout.write("{}\n".format(' '.join(map(lambda z: str(z[1]), output[2]))))
        self.fout.flush()
        #print('{} {} {}\n'.format(output[0], output[1], output[2]))

    def _after_inference(self):
        if self.save_fn:
            self.fout.close()


class LoadedDataFlow(RNGDataFlow):

    def __init__(self, xs, ys, shuffle=True, split='train', do_validation=False):
        assert len(xs) == len(ys)
        self.shuffle = shuffle
        self.dps = [xs, ys]
        n_samples = len(xs)

        self.split = split
        if self.split == 'all':
            self._offset = 0
            self._size = n_samples
        elif self.split == 'train':
            self._offset = 0
            if do_validation:
                self._size = n_samples * 8 // 10
            else:
                self._size = n_samples * 9 // 10
        elif self.split == 'val' or self.split == 'validation':
            if do_validation:
                self._offset = n_samples * 8 // 10
                self._size = n_samples * 9 // 10 - self._offset
            else:
                self._offset = n_samples * 9 // 10
                self._size = n_samples - self._offset

    def size(self):
        return self._size

    def get_data(self):
        idxs = list(range(self._offset, self._offset + self._size))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            #print('{} : {} '.format(k, [dp[k] for dp in self.dps]))
            yield [dp[k] for dp in self.dps]


def get_csv_data(fn, delim=' ', target_dim=-1, num_classes=2):
    if num_classes > 0:
        parse_y = lambda y : int(float(y))
        y_type = int
    else:
        parse_y = lambda y : float(y)
        y_type = np.float32

    with open(fn, 'rt') as fin:
        xs, ys = [], []
        for line in fin:
            line = line.strip().split(delim)
            if target_dim < 0:
                target_dim = len(line) + target_dim
            xs.append(list(map(float, line[0:target_dim] + line[target_dim+1:len(line)])))
            ys.append(parse_y(line[target_dim]))

    xs = np.asarray(xs, dtype=np.float32).reshape([len(xs), -1])
    ys = np.asarray(ys, dtype=y_type)
    return xs, ys, len(xs[0]), num_classes

