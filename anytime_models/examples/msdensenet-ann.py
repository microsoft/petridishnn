import argparse
import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import AnytimeMultiScaleDenseNet
import ann_app_utils 

"""
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_msdensenet_arguments(parser)
    args = parser.parse_args()

    model_cls = AnytimeMultiScaleDenseNet
    args.num_classes = 100 if args.ds_name == 'cifar100' else 10
    args.growth_rate = 6
    args.prediction_feature = 'msdense'
    args.num_scales = 3
    args.reduction_ratio = 0.5

    ann_app_utils.cifar_svhn_train_or_test(args, model_cls)
