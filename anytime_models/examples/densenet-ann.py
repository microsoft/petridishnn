import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger, utils, fs

import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import \
    AnytimeLogDenseNetV1, AnytimeLogDenseNetV2, DenseNet, AnytimeLogLogDenseNet

import ann_app_utils

"""
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_densenet_arguments(parser)
    args = parser.parse_args()

    args.dropout_kp = 0.8
    if args.densenet_version == 'atv1':
        model_cls = AnytimeLogDenseNetV1
    elif args.densenet_version == 'atv2':
        model_cls = AnytimeLogDenseNetV2
    elif args.densenet_version == 'dense':
        model_cls = DenseNet
    elif args.densenet_version == 'loglog':
        model_cls = AnytimeLogLogDenseNet

    ann_app_utils.cifar_svhn_train_or_test(args, model_cls)
