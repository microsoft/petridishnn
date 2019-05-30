#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse

import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import \
    AnytimeLogDenseNetV1, AnytimeLogDenseNetV2, \
    DenseNet, AnytimeLogLogDenseNet

import ann_app_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_densenet_arguments(parser)
    args = parser.parse_args()
    args.ds_name="ilsvrc"
    if args.densenet_version == 'atv1':
        model_cls = AnytimeLogDenseNetV1
    elif args.densenet_version == 'atv2':
        model_cls = AnytimeLogDenseNetV2
    elif args.densenet_version == 'dense':
        model_cls = DenseNet
        args.reduction_ratio = 0.5
    elif args.densenet_version == 'loglog':
        model_cls = AnytimeLogLogDenseNet
   
    assert args.growth_rate == 32, args.growth_rate
    ann_app_utils.train_or_test_ilsvrc(args, model_cls)
