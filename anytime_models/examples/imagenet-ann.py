#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import AnytimeResNet, AnytimeResNeXt

import ann_app_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_resnet_arguments(parser)
    args = parser.parse_args()
    args.ds_name="ilsvrc"
    if args.resnet_version == 'resnet':
        model_cls = AnytimeResNet
    elif args.resnet_version == 'resnext':
        model_cls = AnytimeResNeXt
        args.b_type = 'bottleneck'

    assert args.init_channel == 64, args.init_channel 
    ann_app_utils.train_or_test_ilsvrc(args, model_cls)
