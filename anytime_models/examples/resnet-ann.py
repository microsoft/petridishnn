import argparse

import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import AnytimeResNet, AnytimeResNeXt

import ann_app_utils

"""
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_resnet_arguments(parser)
    args = parser.parse_args()
    if args.resnet_version == 'resnet':
        model_cls = AnytimeResNet
    elif args.resnet_version == 'resnext':
        model_cls = AnytimeResNeXt
        args.b_type = 'bottleneck'
    
    ann_app_utils.cifar_svhn_train_or_test(args, model_cls)
