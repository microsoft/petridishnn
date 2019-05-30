# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import os
import cv2
import multiprocessing

from tensorpack.dataflow import DataFlow, PrefetchDataZMQ, \
    LMDBData, LMDBDataPoint, PrefetchData, \
    MapDataComponent, AugmentImageComponent, BatchData
from tensorpack.dataflow import imgaug, LocallyShuffleData


## ILSVRC mean std info
# the literal value are in rgb. cv2 will read in bgr
ilsvrc_mean = [0.485, 0.456, 0.406][::-1]
ilsvrc_std = [0.229, 0.224, 0.225][::-1]


def get_ilsvrc_augmented_data(subset, options, do_multiprocess=True, is_train=None, shuffle=None):
    input_size = options.input_size if options.input_size else 224
    isTrain = is_train if is_train is not None else (subset == 'train' and do_multiprocess)
    shuffle = shuffle if shuffle is not None else isTrain
    lmdb_path = os.path.join(options.data_dir, 'lmdb2', 'ilsvrc2012_{}.lmdb'.format(subset))
    ds = LMDBData(lmdb_path, shuffle=False)
    if shuffle:
        ds = LocallyShuffleData(ds, 1024*80)  # This is 64G~80G in memory images
    ds = PrefetchData(ds, 1024*8, 1) # prefetch around 8 G
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0) # BGR uint8 data
    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            """
            crop 8%~100% of the original image
            See `Going Deeper with Convolutions` by Google.
            """
            def _augment(self, img, _):
                h, w = img.shape[:2]
                area = h * w
                for _ in range(10):
                    targetArea = self.rng.uniform(0.08, 1.0) * area
                    aspectR = self.rng.uniform(0.75, 1.333)
                    ww = int(np.sqrt(targetArea * aspectR))
                    hh = int(np.sqrt(targetArea / aspectR))
                    if self.rng.uniform() < 0.5:
                        ww, hh = hh, ww
                    if hh <= h and ww <= w:
                        x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                        y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                        out = img[y1:y1 + hh, x1:x1 + ww]
                        out = cv2.resize(
                            out, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                        return out
                out = cv2.resize(
                    img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                return out

        augmentors = [
            Resize(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256),
            imgaug.CenterCrop((input_size, input_size)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    if do_multiprocess:
        ds = PrefetchDataZMQ(ds, min(24, multiprocessing.cpu_count()))
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds



def resnext_training_params(args):
    args.init_lr = 0.1 * (args.batch_size / 256.0)
    args.lr_decay_method = 'exponential'
    args.optimizer = None # use default
    args.lr_decay = 0.1
    args.lr_decay_every = args.max_epoch * 3 // 10
    args.regularize_coef = 'const'
    args.regularize_const = 1e-4
    args.batch_norm_decay = 0.9 ** (args.batch_size / 256.0)
    args.batch_norm_epsilon = 1e-5
    return args

def inception_training_params(args):
    args.optimizer = 'rmsprop'
    args.rmsprop_epsilon = 1.0
    args.init_lr = 0.045 # orig has 50 gpu
    args.lr_decay_method = 'exponential'
    args.lr_decay = 0.94
    args.lr_decay_every = 2. / args.nr_gpu
    args.regularize_coef = 'const'
    args.regularize_const = 1e-4
    args.batch_norm_decay = 0.9997
    args.batch_norm_epsilon = 1e-3
    return args

def zoph_training_params(args):
    args = inception_training_params(args)
    args.init_lr = 0.025 # orig is 0.04 * 50
    args.label_smoothing = 0.1
    args.regularize_const = 4e-5
    # TODO params under tuning:
    # dense dropout, path dropout, batch_norm_params
    #args.dense_dropout_keep_prob = 0.5
    #args.drop_path_keep_prob = 0.7
    return args

def zoph1_training_params(args):
    args = zoph_training_params(args)
    args.init_lr = 0.05
    return args

def zoph2_training_params(args):
    args = zoph_training_params(args)
    args.init_lr = 0.001
    return args

def darts_training_params(args):
    args.optimizer = None
    args.lr_decay_method = 'exponential'
    args.lr_decay = 0.97
    args.lr_decay_every = 1. / args.nr_gpu
    args.init_lr = 0.1
    args.regularize_coef = 'const'
    args.regularize_const = 3e-5
    args.batch_norm_decay = 0.9
    args.batch_norm_epsilon = 1e-5
    return args

def real_training_params(args):
    args = zoph_training_params(args)
    args.rmsprop_epsilon = 0.1
    args.init_lr = 0.1 # ???? paper says 0.001 with 100 gpu
    args.lr_decay = 0.97
    args.regularize_const = 4e-5
    return args

def nasnet_training_params(args):
    args = zoph_training_params(args)
    return args

def training_params_update(args):
    if args.training_type is None or args.training_type == 'resnext':
        args = resnext_training_params(args)
    elif args.training_type == 'darts_imagenet':
        args = darts_training_params(args)
    elif args.training_type == 'zoph':
        args = zoph_training_params(args)
    elif args.training_type == 'zoph1':
        args = zoph_training_params(args)
    elif args.training_type == 'zoph2':
        args = zoph_training_params(args)
    elif args.training_type == 'nasnet':
        args = nasnet_training_params(args)
    elif args.training_type == 'real':
        args = real_training_params(args)
    return args