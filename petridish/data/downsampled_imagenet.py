# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import os
import re
import bisect
import sys
import multiprocessing
import time

from tensorpack.dataflow import DataFlow, PrefetchDataZMQ, \
    LMDBData, LMDBDataPoint, PrefetchData, \
    MapDataComponent, AugmentImageComponent, BatchData
from tensorpack.dataflow import imgaug
from tensorpack.utils import logger

from tensorpack.dataflow.serialize import LMDBSerializer
dump_dataflow_to_lmdb = LMDBSerializer.save


class DownsampledImageNet(DataFlow):

    def __init__(self, data_dir, split, shuffle=True, input_size=None, do_validation=False):
        self.shuffle = shuffle
        if do_validation:
            prefix = 'train_data'
        else:
            prefix = '{}_data'.format(split)

        l_basenames = list(filter(lambda dn : prefix in dn, os.listdir(data_dir)))
        assert len(l_basenames) > 0, "Split {} is empty in {}".format(split, data_dir)
        l_basenames = sorted(l_basenames) # sort by string

        if do_validation:
            n_batches = len(l_basenames)
            n_train_batches = n_batches * 9 // 10
            if split == 'train':
                # effectively, we use 1,10,2,3,..8 for training
                l_basenames = l_basenames[:n_train_batches]
            else:
                # and use 9 for validation
                l_basenames = l_basenames[n_train_batches:]

        # List of dict that has keys
        logger.info("Start loading data...")
        start_time = time.time()
        l_data = [ _load_dictionary(data_dir, basename) for basename in l_basenames]

        x_len = l_data[0]['data'].shape[1]
        computed_input_size = int(np.sqrt(x_len // 3))
        assert input_size is None or computed_input_size == input_size, \
            "inputsize mismatch computed {} , given {}".format(computed_input_size, input_size)
        input_size = computed_input_size
        assert input_size * input_size * 3 == x_len, \
                "Input size {} is not for square image".format(input_size)
        self.input_size = input_size

        def reshape_transpose(mat):
            # The first input_size**2 is red, then G and B. Hence the data are naturally NCHW
            # reshape into NCHW, and then transpose to NHWC for easy pre-processing.
            return mat.reshape([-1, 3, input_size, input_size]).transpose([0, 2, 3, 1])

        self.l_xs = [ reshape_transpose(data['data']) for data in l_data ] # a list of img in NHWC
        self.l_sizes = [xs.shape[0] for xs in self.l_xs] # a list of int
        self.cumsum_sizes = np.cumsum(self.l_sizes)
        self._size = sum(self.l_sizes)
        self.l_ys = [ data['labels'] for data in l_data]  # a list of lists
        logger.info("...done using {} sec".format(time.time() - start_time))
        # NHWC mean image
        mean_img_fn = os.path.join(data_dir, 'mean_img.npz')
        if os.path.exists(mean_img_fn):
            self.mean_img = np.load(mean_img_fn, encoding='bytes')['mean_img']
        else:
            assert split == 'train', "mean_img.npz is not computable on val set"
            self.mean_img = reshape_transpose(sum([ data['mean'] * n  for n, \
                data in zip(self.l_sizes, l_data) ]) / float(self._size))[0]
            self.mean_img = np.asarray(self.mean_img, dtype=np.float32)
            assert len(self.mean_img.shape) == 3 and self.mean_img.shape[2] == 3, \
                "Bug on computing the mean image"
            #self.mean_rgb = self.mean_img.mean(axis=(0,1))
            # store mean_img if it is not exist
            np.savez(mean_img_fn, mean_img=self.mean_img)


    def size(self):
        return self._size


    def get_data(self):
        indices = list(range(self._size))
        if self.shuffle:
            np.random.shuffle(indices)

        for k in indices:
            batch_i = bisect.bisect_right(self.cumsum_sizes, k)
            img_i = k - self.cumsum_sizes[batch_i]
            # We subtract 1 on label because the original label is 1-based
            yield [self.l_xs[batch_i][img_i], self.l_ys[batch_i][img_i] - 1]


def _load_dictionary(data_dir, basename):
    f = np.load(os.path.join(data_dir, basename), encoding='bytes')
    if 'diction' in f:
        return f['diction'].item()
    return f

def _data_batch_dir(data_root, input_size):
    v = sys.version
    if int(v[0]) == 2:
        postfix = '_py2'
    else:
        postfix = ''
    return os.path.join(data_root, 'downsampled_imagenet',
        'imagenet{s}{p}'.format(s=input_size, p=postfix))


class ImageNet16(DownsampledImageNet):

    def __init__(self, data_root, split, shuffle=True, do_validation=False):
        super(ImageNet16, self).__init__(_data_batch_dir(data_root, 16),
            split, shuffle=shuffle, input_size=16, do_validation=do_validation)


class ImageNet32(DownsampledImageNet):

    def __init__(self, data_root, split, shuffle=True, do_validation=False):
        super(ImageNet32, self).__init__(_data_batch_dir(data_root, 32),
            split, shuffle=shuffle, input_size=32, do_validation=do_validation)


class ImageNet64(DownsampledImageNet):

    def __init__(self, data_root, split, shuffle=True, do_validation=False):
        super(ImageNet64, self).__init__(_data_batch_dir(data_root, 64),
            split, shuffle=shuffle, input_size=64, do_validation=do_validation)


def is_ds_name_downsampled_imagenet(ds_name):
    reret = re.search(r'^imagenet([0-9]*)$', ds_name)
    return reret is not None


def ds_name_to_input_size(ds_name):
    reret = re.search(r'^imagenet([0-9]*)$', ds_name)
    return int(reret.group(1))


def get_downsampled_imagenet_augmented_data(subset, options,
        do_multiprocess=True, do_validation=False, shuffle=None):
    isTrain = subset == 'train' and do_multiprocess
    shuffle = shuffle if shuffle is not None else isTrain

    reret = re.search(r'^imagenet([0-9]*)$', options.ds_name)
    input_size = int(reret.group(1))

    ds = DownsampledImageNet(_data_batch_dir(options.data_dir, input_size),\
         subset, shuffle, input_size, do_validation=do_validation)

    pp_mean = ds.mean_img
    paste_size = ds.input_size * 5 // 4
    crop_size = ds.input_size
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((paste_size, paste_size)),
            imgaug.RandomCrop((crop_size, crop_size)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    if do_multiprocess:
        ds = PrefetchData(ds, 4, 2)
    return ds


def _convert_pickle_3to2(src_fn, dest_fn, split=1):
    # Have to run this in python 3
    d = np.load(src_fn, encoding='bytes')
    if split > 1:
        size = d['data'].shape[0]
        d_tmp = dict()
        start = 0
        step = size // split
        end = step
        for si in range(split):
            d_tmp['data'] = d['data'][start:end]
            d_tmp['labels'] = d['labels'][start:end]
            if 'mean' in d.keys():
                d_tmp['mean'] = d['mean']
            start = end
            end += step
            if si + 1 == split:
                end = size
            fn = dest_fn + '_{}'.format(si)
            np.savez(dest_fn + '_{}'.format(si), diction=d_tmp)
            #with open(dest_fn + '_{}'.format(si), 'wb') as fout:
            #    pickle.dump(d_tmp, fout, protocol=2)
            os.rename(fn + '.npz', fn)

    else:
        #with open(dest_fn, 'wb') as fout:
        #    pickle.dump(d, fout, protocol=2)
        np.savez(dest_fn, diction=d)
        os.rename(dest_fn + '.npz', dest_fn)

def _downsampled_imagenet_3to2(src_dn, dest_dn, split=1):
    assert os.path.exists(src_dn) and os.path.exists(dest_dn), 'src and dest must exist'
    l_basenames = os.listdir(src_dn)
    for basename in l_basenames:
        reret = re.search(r'train_data_batch_([0-9]*)', basename)
        if basename == 'val_data' or reret:
            src_fn = os.path.join(src_dn, basename)
            dest_fn = os.path.join(dest_dn, basename)
            _convert_pickle_3to2(src_fn, dest_fn, split)


if __name__ == '__main__':
    # Convert from python3 pickle to python2 pickle b/c docker currently is python2 only.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--split', type=int, default=1)
    args = parser.parse_args()
    src = args.src
    dest = args.dest
    split = args.split

    assert os.path.exists(src), 'src doesn\'t exists'
    assert os.path.isdir(src), 'src must be a dir of py3 batches'
    _downsampled_imagenet_3to2(src, dest, split)

    import re
    reret = re.search(r'[a-z]*([0-9]*)$', src)
    ds = DownsampledImageNet(dest,
        'train', True, int(reret.group(1)), do_validation=False)