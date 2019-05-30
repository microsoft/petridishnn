import numpy as np
import os
import re
import bisect
import glob
from scipy.io import wavfile
import sys
import multiprocessing
import itertools
import math, random
import functools

from tensorpack.dataflow import DataFlow, PrefetchDataZMQ, \
    PrefetchData, MapDataComponent, MapData, BatchData
from tensorpack.dataflow.serialize import LMDBSerializer
dump_dataflow_to_lmdb = LMDBSerializer.save

def _load_list_in_file(fn):
    with open(fn, 'rt') as fin:
        lines = fin.read()
    return sorted(lines.strip().split())

def _is_fn_in_ll_fn(fn, ll_fn):
    for l_fn in ll_fn:
        idx = bisect.bisect_left(l_fn, fn)
        if l_fn[idx] == fn:
            return True
    return False

def _full_fn_to_record_fn(full_fn):
    """

    Returns:
    str : tree/c24d96eb_nohash_0.wav
    """
    return os.path.join(os.path.basename(os.path.dirname(full_fn)), os.path.basename(full_fn))


ALL_WORDS = sorted(['bed', 'cat', 'down', 'five', 'forward', 'go', 'house', 'left', 'marvin', 'no',
    'on', 'seven', 'six', 'tree',  'up', 'visual', 'yes', 'backward', 'bird', 'dog', 'eight', 'follow',
    'four', 'happy', 'learn', 'nine', 'off', 'one', 'right', 'sheila', 'stop', 'three', 'two', 'wow', 'zero'])


DEFAULT_TRAIN_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185
SAMPLE_RATE = 16000
DESIRED_SAMPLES = 16000


class RawSpeechCommandsData(object):

    def __init__(self, data_dir, split, train_words=None):
        """
        """
        test_list_fn = os.path.join(data_dir, 'testing_list.txt')
        val_list_fn = os.path.join(data_dir, 'validation_list.txt')
        test_list = _load_list_in_file(test_list_fn)
        val_list = _load_list_in_file(val_list_fn)

        if train_words is None:
            train_words = DEFAULT_TRAIN_WORDS

        l_background_noises = glob.glob(os.path.join(data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav'))
        l_background_noises = list(map(_full_fn_to_record_fn, l_background_noises))

        if split in ['validation', 'val']:
            self.split = 'val'
            l_fn = val_list
        elif split in ['testing', 'test']:
            self.split = 'test'
            l_fn = test_list
        elif split in ['train', 'training']:
            self.split = 'train'
            train_list = []
            for word in ALL_WORDS:
                train_list.extend(map(_full_fn_to_record_fn, glob.glob(os.path.join(data_dir, word, '*.wav'))))
            l_fn = train_list
        else:
            raise ValueError("Unknown split name {}".format(split))

        self.l_fn = l_fn
        self.l_background_noises = l_background_noises
        self._size = len(self.l_fn)
        self.data_dir = data_dir
        self.data = None

    def size(self):
        return self._size

    def load(self):
        if self.data is not None:
            return
        self.data = []
        self.labels = []
        for fn in self.l_fn:
            fps, x = wavfile.read(os.path.join(self.data_dir, fn))
            if fps != SAMPLE_RATE:
                print("MEOW!!!!! different frame rate")
            label = os.path.basename(os.path.dirname(fn))
            self.data.append(x)
            self.labels.append(label)

        if self.split == 'train':
            self.noises = []
            for fn in self.l_background_noises:
                fps, x = wavfile.read(os.path.join(self.data_dir, fn))
                if fps != SAMPLE_RATE:
                    print("MEOW!!!!! different frame rate")
                self.noises.append(x)

    def save_to_pickles(self, dest_dir, n_batch=1):
        self.load()
        if self.split == 'train':
            np.savez(os.path.join(dest_dir, BACKGROUND_NOISE_DIR_NAME), noises=self.noises)

        N = len(self.data)
        start = 0
        step = N // n_batch
        end = step
        for si in range(n_batch):
            if si + 1 == n_batch:
                end = N
            postfix = ""
            if n_batch > 1:
                postfix = "_batch_{}".format(si + 1)
            batch_name = self.split + '_data' + postfix
            np.savez(os.path.join(dest_dir, batch_name), \
                data=self.data[start:end], labels=self.labels[start:end])
            start = end
            end += step


def convert_to_npz():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dest_dir', type=str, default=None)
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        data = RawSpeechCommandsData(split=split, data_dir=args.data_dir)
        if split == 'train':
            n_batch = 4
        else:
            n_batch = 1
        data.save_to_pickles(dest_dir=args.dest_dir, n_batch=n_batch)


class SpeechCommandsDataFlow(DataFlow):

    def __init__(self, data_dir, split, shuffle=True, train_words=None,
            silence_percentage=10., unknown_percentage=10.0):
        self.shuffle = shuffle
        train_words = DEFAULT_TRAIN_WORDS if train_words is None else train_words
        self.train_words = train_words
        self.split = split

        words_to_label = dict()
        words_to_label[SILENCE_LABEL] = SILENCE_INDEX
        words_to_label[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX
        offset = 2
        for wi, word in enumerate(train_words):
            words_to_label[word] = wi + offset
        for word in ALL_WORDS:
            if word in train_words:
                continue
            words_to_label[word] = UNKNOWN_WORD_INDEX

        l_npz = glob.glob(os.path.join(data_dir, '{}_data*.npz'.format(split)))
        l_data = [ np.load(npz, encoding='bytes')  for npz in l_npz]
        l_xs = [ list(data['data'])   for data in l_data ]
        l_ys = [ list(map(lambda ystr : words_to_label[str(ystr)],  data['labels']))  \
            for data in l_data]

        # filter the xs and ys into train_words and backgrounds.
        self.xs = []
        self.ys = []
        unknown_xs = []
        for xs, ys in zip (l_xs, l_ys):
            is_background = list(map(lambda y : y == UNKNOWN_WORD_INDEX or y == SILENCE_INDEX, ys))
            self.xs.extend(itertools.compress(xs, map(lambda t : not t, is_background)))
            self.ys.extend(itertools.compress(ys, map(lambda t : not t, is_background)))
            unknown_xs.extend(itertools.compress(xs, is_background))

        # compute unknown and silence sizes
        set_size = len(self.xs)
        unknown_size = int(math.ceil(set_size * unknown_percentage / 100.0))
        silence_size = int(math.ceil(set_size * silence_percentage / 100.0))

        # select backgrounds based on percentage
        random.seed(RANDOM_SEED)
        random.shuffle(unknown_xs)
        self.xs.extend(unknown_xs[:unknown_size])
        self.ys.extend([UNKNOWN_WORD_INDEX  for _ in range(unknown_size)])

        # add silece based on percentage
        self.xs.extend([np.zeros([SAMPLE_RATE], dtype=np.int16) for _ in range(silence_size)])
        self.ys.extend([SILENCE_INDEX for _ in range(silence_size) ])

        self._size = len(self.xs)
        self.dps = [self.xs, self.ys]

        if split == 'train':
            self.noises = list(np.load(os.path.join(data_dir, BACKGROUND_NOISE_DIR_NAME + '.npz'),
                                       encoding='bytes')['noises'])


    def size(self):
        return self._size


    def get_data(self):
        indices = list(range(self._size))
        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            yield [ dp[idx] for dp in self.dps ]


def _to_float(x):
    assert x.dtype == np.int16
    return x.astype(np.float32) / np.iinfo(np.int16).max


def _pad_or_clip_to_desired_sample(x):
    len_x = len(x)
    if len_x == DESIRED_SAMPLES:
        return x
    extra = len_x - DESIRED_SAMPLES
    if extra > 0:
        start = extra // 2
        end = - extra + start
        return x[start:end]
    pad = -extra
    pleft = pad // 2
    return np.pad(x, [pleft, pad - pleft], mode='constant')


def _time_shift(x):
    assert x.dtype == np.float32
    time_shift_ms = 100. # in ms
    time_shift = int(time_shift_ms / len(x) * 1000)
    if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
    else:
        time_shift_amount = 0
    if time_shift_amount > 0:
        x = np.pad(x[:-time_shift_amount], [time_shift_amount, 0], mode='constant')
    elif time_shift_amount < 0:
        x = np.pad(x[-time_shift_amount:], [0, -time_shift_amount], mode='constant')
    return x


def _add_noise(d, noises):
    """
    x (1-d array) : float array
    y (int) : label index
    """
    x, y = d
    assert x.dtype == np.float32
    desired_samples = DESIRED_SAMPLES
    background_frequency = 0.8
    background_volume_range = 0.1

    # random sample noise
    if y == SILENCE_INDEX:
        background_index = np.random.randint(len(noises))
        background_samples = noises[background_index]
        if len(background_samples) <= desired_samples:
            raise ValueError(
                'Background sample is too short! Need more than %d'
                ' samples but only %d were found' %
                (desired_samples, len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - desired_samples)
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples])
        if y == SILENCE_INDEX:
            background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
            background_volume = np.random.uniform(0, background_volume_range)
        else:
            background_volume = 0
    else:
        background_reshaped = np.zeros([desired_samples])
        background_volume = 0

    # Merge noise with data
    x = np.clip(x + background_volume * background_reshaped, -1.0, 1.0)
    return [x, y]


def get_augmented_speech_commands_data(subset, options,
        do_multiprocess=True, shuffle=True):
    isTrain = subset == 'train' and do_multiprocess
    shuffle = shuffle if shuffle is not None else isTrain

    ds = SpeechCommandsDataFlow(os.path.join(options.data_dir, 'speech_commands_v0.02'),
        subset, shuffle, None)
    if isTrain:
        add_noise_func = functools.partial(_add_noise, noises=ds.noises)
    ds = MapDataComponent(ds, _pad_or_clip_to_desired_sample, index=0)
    ds = MapDataComponent(ds, _to_float, index=0)
    if isTrain:
        ds = MapDataComponent(ds, _time_shift, index=0)
        ds = MapData(ds, add_noise_func)
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    if do_multiprocess:
        ds = PrefetchData(ds, 4, 4)
    return ds


def _test_get_augmented_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    subset, shuffle = 'train', True
    args.batch_size = 100
    args.nr_gpu = 1
    ds = get_augmented_speech_commands_data(subset, args, True, shuffle)
    return ds

if __name__ == '__main__':
    #convert_to_npz()
    ret = _test_get_augmented_data()
