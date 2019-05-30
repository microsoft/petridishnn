import numpy as np
import multiprocessing

from tensorpack.utils import logger
from tensorpack.dataflow import (PrefetchData,
    AugmentImageComponent, BatchData)
from tensorpack.dataflow import imgaug, dataset
from petridish.data.misc import Cutout


def get_cifar_augmented_data(
        subset, options, do_multiprocess=True, do_validation=False, shuffle=None):
    isTrain = subset == 'train' and do_multiprocess
    shuffle = shuffle if shuffle is not None else isTrain
    if options.num_classes == 10 and options.ds_name == 'cifar10':
        ds = dataset.Cifar10(subset, shuffle=shuffle, do_validation=do_validation)
        cutout_length = 16
        n_holes=1
    elif options.num_classes == 100 and options.ds_name == 'cifar100':
        ds = dataset.Cifar100(subset, shuffle=shuffle, do_validation=do_validation)
        cutout_length = 8
        n_holes=1
    else:
        raise ValueError('Number of classes must be set to 10(default) or 100 for CIFAR')
    logger.info('{} set has n_samples: {}'.format(subset, len(ds.data)))
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        logger.info('Will do cut-out with length={} n_holes={}'.format(
            cutout_length, n_holes
        ))
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
            Cutout(length=cutout_length, n_holes=n_holes),
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