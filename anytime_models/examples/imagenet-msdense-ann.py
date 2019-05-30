#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse

import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import AnytimeMultiScaleDenseNet

import ann_app_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_msdensenet_arguments(parser)
    args = parser.parse_args()

    model_cls = AnytimeMultiScaleDenseNet
    # Fixed parameter that are only for msdense
    args.ds_name="ilsvrc"
    args.num_classes == 1000
    args.growth_rate=16
    if hasattr(args, 'msdensenet_depth') and args.msdensenet_depth is not None:
        args.stack = (args.msdensenet_depth - 3) // 5
    else:
        assert hasattr(args, 'block_config') and args.block_config is not None
    args.prediction_feature='msdense'
    args.num_scales=4
    args.reduction_ratio = 0.5
    args.b_type = 'bottleneck'
    args.s_type = 'imagenet'

    ann_app_utils.train_or_test_ilsvrc(args, model_cls)
