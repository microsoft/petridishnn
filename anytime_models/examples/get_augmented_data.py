import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorpack import *
from tensorpack.dataflow.dataset import map_cityscape_labels_to_train_labels


## ILSVRC mean std info
# the literal value are in rgb. cv2 will read in bgr
ilsvrc_mean = [0.485, 0.456, 0.406][::-1] 
ilsvrc_std = [0.229, 0.224, 0.225][::-1]


## Cityscapes mean std info
# std was not actually computed, just some reasonable number
cityscapes_mean = [ 72.02736664,  81.89767456,  71.22579956]  # From 100 imgs
cityscapes_std = [96., 96., 96.]  # From 100 imgs: [ 45.83293352,  46.73820686,  45.81782445]

def get_distill_target_data(subset, options):
    distill_target_fn = os.path.join(options.data_dir, 'distill_targets', 
        '{}_distill_target_{}.bin'.format(options.ds_name, subset))
    ds = BinaryData(distill_target_fn, options.num_classes)
    return ds

def join_distill_and_shuffle(ds, subset, options, buffer_size=None):
    ds_distill = get_distill_target_data(subset, options)
    ds = JoinData([ds, ds_distill])
    if buffer_size is None:
        buffer_size = ds.size()
    ds = LocallyShuffleData(ds, buffer_size)
    return ds

def get_cifar_augmented_data(subset, options, do_multiprocess=True, do_validation=False, shuffle=None):
    isTrain = subset == 'train' and do_multiprocess
    use_distill = isTrain and options.alter_label
    shuffle = shuffle if shuffle is not None else (isTrain and not options.alter_label)
    if options.num_classes == 10:
        ds = dataset.Cifar10(subset, shuffle=shuffle, do_validation=do_validation)
    elif options.num_classes == 100:
        ds = dataset.Cifar100(subset, shuffle=shuffle, do_validation=do_validation)
    else:
        raise ValueError('Number of classes must be set to 10(default) or 100 for CIFAR')
    logger.info('{} set has n_samples: {}'.format(subset, len(ds.data)))
    pp_mean = ds.get_per_pixel_mean()
    if use_distill:
        ds = join_distill_and_shuffle(ds, subset, options)
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
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
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_svhn_augmented_data(subset, options, do_multiprocess=True, shuffle=None):
    isTrain = subset == 'train' and do_multiprocess
    use_distill = isTrain and options.alter_label
    shuffle = shuffle if shuffle is not None else (isTrain and not options.alter_label)
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train', shuffle=shuffle)
        d2 = dataset.SVHNDigit('extra', shuffle=shuffle)
        if use_distill:
            d1 = join_distill_and_shuffle(d1, 'train', options)
            d2 = join_distill_and_shuffle(d2, 'extra', options)
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit(subset, shuffle=shuffle)

    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.Brightness(10),
            imgaug.Contrast((0.8, 1.2)),
            imgaug.GaussianDeform(  # this is slow. without it, can only reach 1.9% error
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (40, 40), 0.2, 3),
            imgaug.RandomCrop((32, 32)),
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    if do_multiprocess:
        ds = PrefetchData(ds, 5, 5)
    return ds


def get_ilsvrc_augmented_data(subset, options, do_multiprocess=True, is_train=None, shuffle=None):
    isTrain = is_train if is_train is not None else (subset == 'train' and do_multiprocess)
    shuffle = shuffle if shuffle is not None else isTrain
    use_distill = isTrain and options.alter_label
    lmdb_path = os.path.join(options.data_dir, 'lmdb2', 'ilsvrc2012_{}.lmdb'.format(subset))
    ds = LMDBData(lmdb_path, shuffle=False)
    if isTrain and use_distill:
        fn = os.path.join(options.data_dir, 'distill_targets', 
            '{}_distill_target_{}.bin'.format(options.ds_name, subset))
        dstl = BinaryData(fn, options.num_classes)
        ds = JoinData([ds, dstl])
    if shuffle:
        ds = LocallyShuffleData(ds, 1024*64)  # This is 64G~80G in memory images
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
                        out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
                        return out
                out = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
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
            imgaug.CenterCrop((224, 224)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    if do_multiprocess:
        ds = PrefetchDataZMQ(ds, min(24, multiprocessing.cpu_count()))
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds


def get_pascal_voc_augmented_data(subset, options, do_multiprocess=True):
    side = 224
    n_classes = 21 # include 0 : background ; 21 : void
    isTrain = subset[:5] == 'train' and do_multiprocess
    lmdb_path = os.path.join(options.data_dir, 'pascal_voc_lmdb', 
        'pascal_voc2012_{}.lmdb'.format(subset))
    ds = LMDBData(lmdb_path, shuffle=False)
    if isTrain:
       ds = LocallyShuffleData(ds, 1024*7)
    ds = PrefetchData(ds, 1024*7, 1)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: x[0].astype(np.float32), 0)

    def one_hot(y_img, n_classes=n_classes, last_is_void=True):
        assert len(y_img.shape) == 2
        y_img[y_img >= n_classes] = n_classes
        h, w = y_img.shape
        y_one_hot = np.eye(n_classes + int(last_is_void), dtype=np.float32)
        y_one_hot = y_one_hot[y_img.astype(int).reshape([-1])].reshape([h,w,-1])
        if last_is_void:
            y_one_hot = y_one_hot[:,:,:-1]
        return y_one_hot
        
    ds = MapDataComponent(ds, lambda y: one_hot(y[0]), 1)

    pascal_voc_mean = dataset.PascalVOC.mean
    pascal_voc_std = dataset.PascalVOC.std
    x_augmentors = [
        imgaug.MapImage(lambda x: (x - pascal_voc_mean)/pascal_voc_std),
    ]
    if isTrain:
        xy_augmentors = [
            imgaug.RotationAndCropValid(max_deg=10),
            imgaug.Flip(horiz=True),
            imgaug.GaussianBlur(max_size=3),
            imgaug.ResizeShortestEdge(256),
            imgaug.RandomCrop((side, side)),
        ]
    else:
        xy_augmentors = [
            imgaug.ResizeShortestEdge(256),
            imgaug.CenterCrop((side, side)),
        ]
    if len(xy_augmentors) > 0:
        ds = AugmentImageComponents(ds, xy_augmentors, copy=False)
    if len(x_augmentors) > 0:
        ds = AugmentImageComponent(ds, x_augmentors, copy=False)
    if do_multiprocess:
        ds = PrefetchData(ds, 5, 5)
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds


def get_cityscapes_augmented_data(subset, options, do_multiprocess=True):
    """
    Note that the original cityscapes dataset contains more classes than we actually evaluate on.
    We remap the original class ids to new 'training ids'.  This needs to be remapped back if we
    want to use any kind of standard evaluation code for cityscapes (and needs to be remembered
    when we're looking up class names!).  You can find the mapping in
    dataflow/dataset/cityscapes_labels.py
    """
    side = 224
    n_classes = 19  # include 0 : background ; -1 : license plate
    isTrain = subset[:5] == 'train' and do_multiprocess
    lmdb_path = os.path.join(options.data_dir, 'cityscapes_lmdb',
                             'cityscapes_{}.lmdb'.format(subset))
    ds = LMDBData(lmdb_path, shuffle=False)
    if isTrain:
        ds = LocallyShuffleData(ds, 2048)
    ds = PrefetchData(ds, 1024 * 8, 1)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: x[0].astype(np.float32), 0)

    def one_hot(y_img, n_classes=n_classes):
        # We map all non-evaluated classes onto the 'void' class
        # (label = n_classes)
        assert len(y_img.shape) == 2
        map_cityscape_labels_to_train_labels(y_img)
        y_img[y_img >= n_classes] = n_classes
        h, w = y_img.shape
        y_one_hot = np.eye(n_classes + 1, dtype=np.float32)
        y_one_hot = y_one_hot[y_img.astype(int).reshape([-1])].reshape([h,w,-1])
        y_one_hot = y_one_hot[:,:,:-1]
        return y_one_hot

    ds = MapDataComponent(ds, lambda y: one_hot(y[0]), 1)

    x_augmentors = [
        imgaug.MapImage(lambda x: (x - cityscapes_mean)/cityscapes_std),
    ]
    if isTrain:
        xy_augmentors = [
            imgaug.RotationAndCropValid(max_deg=10),
            imgaug.Flip(horiz=True),
            imgaug.GaussianBlur(max_size=3),
            imgaug.ResizeShortestEdge(256),
            imgaug.RandomCrop((side, side)),
        ]
    else:
        xy_augmentors = [
            imgaug.ResizeShortestEdge(256),
            imgaug.CenterCrop((side, side)),
        ]
    if len(xy_augmentors) > 0:
        ds = AugmentImageComponents(ds, xy_augmentors, copy=False)
    if len(x_augmentors) > 0:
        ds = AugmentImageComponent(ds, x_augmentors, copy=False)
    if do_multiprocess:
        ds = PrefetchData(ds, 5, 5)
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds
