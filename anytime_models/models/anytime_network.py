import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils import anytime_loss, logger, utils, fs

from collections import namedtuple
import bisect

from anytime_models.models.online_learn import Exp3CPU, RWMCPU, \
        FixedDistributionCPU, ThompsonSamplingCPU, AdaptiveLossWeight

# Best choice for samloss for AANN if running anytime networks.
BEST_AANN_METHOD=6
# method id for not using AANN
NO_AANN_METHOD=0

# func type for computing optimal at options.opt_at see anytime_loss.loss_weights
FUNC_TYPE_OPT = 2
# func type for computing ANN/AANN
FUNC_TYPE_ANN = 5

# SAMLOSS/ls_method for adaloss
ADALOSS_LS_METHOD=100

"""
    cfg is a tuple that contains
    ([ <list of n_units per block], <b_type>, <start_type>)

    n_units_per_block is a list of int
    b_type is in ["basic", "bottleneck"]
    start_type is in ["basic", "imagenet"]
"""
NetworkConfig = namedtuple('Config', ['n_units_per_block', 'b_type', 's_type'])

def compute_cfg(options):
    b_type = options.b_type
    s_type = options.s_type
    if hasattr(options, 'block_config') and options.block_config is not None:
        assert len(options.block_config) > 0
        n_units_per_block = list(map(int, options.block_config.strip().split(',')))

    elif hasattr(options, 'depth') and options.depth is not None:
        if options.depth == 18:
            n_units_per_block = [2,2,2,2]
            b_type = 'basic'
        elif options.depth == 34:
            n_units_per_block = [3,4,6,3]
            b_type = 'basic'
        elif options.depth == 50:
            n_units_per_block = [3,4,6,3]
            b_type = 'bottleneck'
        elif options.depth == 101:
            n_units_per_block = [3,4,23,3]
            b_type = 'bottleneck'
        elif options.depth == 152:
            n_units_per_block = [3,8,36,3]
            b_type = 'bottleneck'
        elif options.depth == 26:
            n_units_per_block = [2,2,2,2]
            b_type = 'bottleneck'
        elif options.depth == 14:
            n_units_per_block = [1,1,1,1]
            b_type = 'bottleneck'
        else:
            raise ValueError('depth {} must be in [18, 34, 50, 101, 152, 26, 14]'\
                .format(options.depth))
        s_type = 'imagenet'

    elif hasattr(options, 'densenet_depth') and options.densenet_depth is not None:
        if options.densenet_depth == 121:
            n_units_per_block = [6, 12, 24, 16] # 58; s = 9
        elif options.densenet_depth == 169:
            n_units_per_block = [6, 12, 32, 32] # 82; s = 14
        elif options.densenet_depth == 201:
            n_units_per_block = [6, 12, 48, 32] # 98; s = 17
        elif options.densenet_depth == 265:
            n_units_per_block = [6, 12, 64, 48] # 130; s = 24
        elif options.densenet_depth == 63:
            n_units_per_block = [3, 6, 12, 8] # 29 ; s = 5
        elif options.densenet_depth == 35:
            n_units_per_block = [2, 3, 6, 4]  # 15 ; s = 3
        elif options.densenet_depth == 197:
            n_units_per_block = [16, 16, 32, 32]
        elif options.densenet_depth == 217:
            n_units_per_block = [6, 12, 56, 32]
        elif options.densenet_depth == 229:
            n_units_per_block = [16, 16, 48, 32]
        elif options.densenet_depth == 369:
            n_units_per_block = [8, 16, 80, 80] # 184; s = 35
        elif options.densenet_depth == 409:
            n_units_per_block = [6, 12, 120, 64]
        elif options.densenet_depth == 205:
            n_units_per_block = [6, 12, 66, 16]
        else:
            raise ValueError('densenet depth {} is undefined'\
                .format(options.densenet_depth))
        b_type = 'bottleneck'
        s_type = 'imagenet'

    elif hasattr(options, 'msdensenet_depth') and options.msdensenet_depth is not None:
        if options.msdensenet_depth == 38:
            n_units_per_block = [10,9,10,9] #[9, 10, 9, 9]
            s_type = 'imagenet'
            # g = 16
            # s = 7
        elif options.msdensenet_depth == 33:
            n_units_per_block = [9,8,8,8] #[8, 8, 8, 8]
            s_type = 'imagenet'
            # s = 6
        elif options.msdensenet_depth == 23:
            n_units_per_block = [6,6,5,6] #[5, 6, 6, 5]
            s_type = 'imagenet'
            # s = 4
        elif options.msdensenet_depth == 15:
            n_units_per_block = [5,5,5]
            s_type = 'basic'
        elif options.msdensenet_depth == 18:
            n_units_per_block = [6,6,6]
            s_type = 'basic'
        elif options.msdensenet_depth == 24:
            n_units_per_block = [8,8,8]
            s_type = 'basic'
        elif options.msdensenet_depth == 27:
            n_units_per_block = [9,9,9]
            s_type = 'basic'
        elif options.msdensenet_depth == 45:
            n_units_per_block = [15,15,15]
            s_type = 'basic'
        else:
            raise ValueError('Undefined msdensenet_depth')
        b_type = 'bottleneck'

    elif hasattr(options, 'fcdense_depth') and options.fcdense_depth is not None:
        if options.densenet_version in ['atv1', 'loglog', 'atv2', 'dense']:
            if options.fcdense_depth == 103:
                n_units_per_block = [ 4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4 ]
            else:
                raise ValueError('FC dense net depth {} is undefined'\
                    .format(options.fcdense_depth))
            b_type = 'basic'
            s_type = 'basic'

        elif options.densenet_version == 'c2f':
            if options.fcdense_depth == 22:
                n_units_per_block = [5, 5, 5, 6]
            elif options.fcdense_depth == 25:
                n_units_per_block = [9, 7, 5, 3]
            elif options.fcdense_depth == 33:
                n_units_per_block = [8, 8, 8, 8]
            elif options.fcdense_depth == 38:
                n_units_per_block = [10, 9, 10, 9]
            elif options.fcdense_depth == 40:
                n_units_per_block = [10, 10, 10, 10]
            else:
                raise ValueError('FCN coarse2fine depth {} is undefined'.format(options.c2f_depth))
            b_type = 'bottleneck'
            s_type = 'imagenet'

    elif options.num_units is not None:
        #option.n is set
        n_units_per_block = [options.num_units]*options.n_blocks

    config = NetworkConfig(n_units_per_block, b_type, s_type)
    logger.info('The finalized network config is {}'.format(config))
    return config


def parser_add_common_arguments(parser):
    """
        Parser augmentation for anytime resnet/common
    """
    # special group that handles the network depths
    # For each networ type, add its special arg name here I guess.
    depth_group = parser.add_mutually_exclusive_group(required=True)
    depth_group.add_argument('--block_config', help='Number of units per block in a '\
                        +'comma-separated list. This OVERWRITES other depth info.',
                        type=str)
    depth_group.add_argument('-n', '--num_units',
                            help='number of units in each stage',
                            type=int)
    # network complexity
    parser.add_argument('--n_blocks', help='Number of residual blocks, don\'t change usually.'
                        +' Only used if num_units is set',
                        type=int, default=3)
    parser.add_argument('-w', '--width',
                        help='(Legacy) the width of the anytime layer grid. Set to 1',
                        type=int, default=1, choices=[1])
    parser.add_argument('-c', '--init_channel',
                        help='number of channels at the start of the network',
                        type=int, default=16)
    parser.add_argument('-s', '--stack',
                        help='number of units per stack, '
                        +'i.e., number of units per prediction, or prediction period',
                        type=int, default=1)
    parser.add_argument('--min_predict_unit', help='Min unit idx for anytime prediction, 0 based',
                        type=int, default=0)
    parser.add_argument('--weights_at_block_ends',
                        help='Whether only have weights>0 at block ends, useful for fcn',
                        default=False, action='store_true')
    parser.add_argument('--s_type', help='starting conv type',
                        type=str, default='basic', choices=['basic', 'imagenet'])
    parser.add_argument('--b_type', help='block type',
                        type=str, default='basic', choices=['basic', 'bottleneck'])
    parser.add_argument('--prediction_feature',
                        help='Type of feature processing for prediction',
                        type=str, default='none', choices=['none', '1x1', 'msdense', 'bn', 'rescale'])
    parser.add_argument('--prediction_feature_ch_out_rate',
                        help='ch_out= int( <rate> * ch_in)',
                        type=np.float32, default=1.0)
    parser.add_argument('--num_predictor_copies',
                        help='Number of copies of predictors at each exit',
                        type=int, default=1)

    ## alternative_training_target, distillation/compression
    parser.add_argument('--alter_label', help="Type of alternative target to use",
                        default=False, action='store_true')
    parser.add_argument('--alter_loss_w', help="percentage of alter loss weight",
                        type=np.float32, default=0.5)
    parser.add_argument('--alter_label_activate_frac',
                        help="Fraction of anytime predictions that uses alter_label",
                        type=np.float32, default=0.75)
    parser.add_argument('--high_temperature', help='Temperature for training distill targets',
                        type=np.float32, default=1.0)

    ## stop gradient / forward thinking / boost-net / no-grad
    parser.add_argument('--stop_gradient', help='Whether to stop gradients.',
                        default=False, action='store_true')
    parser.add_argument('--sg_gamma', help='Gamma for partial stop_gradient',
                        type=np.float32, default=0)

    ## selecting loss (aka ls_method, samloss)
    parser.add_argument('--init_select_idx', help='the loss anytime_idx to select initially',
                        type=int)
    parser.add_argument('--samloss',
                        help='Method to Sample losses to update',
                        type=int, default=6)
    parser.add_argument('--exp_gamma', help='Gamma for exp3 in sample loss',
                        type=np.float32, default=0.3)
    parser.add_argument('--adaloss_gamma', help='Gamma for adaloss',
                        type=np.float32, default=0.07)
    parser.add_argument('--adaloss_momentum', help='Adaloss momentum',
                        type=np.float32, default=0.9)
    parser.add_argument('--adaloss_update_per', help='Adaloss update weights every number of iter',
                        type=int, default=100)
    parser.add_argument('--adaloss_final_extra', help='Adaloss up-weights the final loss',
                        type=np.float32, default=0.0)
    parser.add_argument('--sum_rand_ratio', help='frac{Sum weight}{randomly selected weight}',
                        type=np.float32, default=2.0)
    parser.add_argument('--last_reward_rate', help='rate of last reward in comparison to the max',
                        type=np.float32, default=0.85)
    parser.add_argument('--is_select_arr', help='Whether select_idx is an int or an array of float',
                        default=False, action='store_true')
    parser.add_argument('--adaloss_order', help='Zero-th or first order adaloss',
                        type=int, default=0, choices=[0,1])

    ## loss_weight computation
    parser.add_argument('-f', '--func_type',
                        help='Type of non-linear spacing to use: 0 for exp, 1 for sqr',
                        type=int, default=FUNC_TYPE_ANN)
    parser.add_argument('--exponential_base', help='Exponential base',
                        type=np.float32)
    parser.add_argument('--opt_at', help='Optimal at',
                        type=int, default=-1)
    parser.add_argument('--last_weight_to_early_sum',
                        help='Final prediction  weight divided by sum of early weights',
                        type=np.float32, default=1.0)
    parser.add_argument('--normalize_weights',
                        help='method to normalize the weights.'\
                        +' last: last one will have 1. all : sum to 1. log : sum to log(N)'\
                        +' Last seems to work the best. log for back-compatibility for f=5,9,10',
                        type=str, default='last', choices=['last', 'all', 'log'])

    ## misc: training params, data-set params, speed/memory params
    parser.add_argument('--init_lr', help='The initial learning rate',
                        type=np.float32, default=0.01)
    parser.add_argument('--sgd_moment', help='moment decay for SGD',
                        type=np.float32, default=0.9)
    parser.add_argument('--batch_norm_decay', help='decay rate of batchnorms',
                        type=np.float32, default=0.9)
    parser.add_argument('--num_classes', help='Number of classes',
                        type=int, default=10)
    parser.add_argument('--regularize_coef', help='How coefficient of regularization decay',
                        type=str, default='const', choices=['const', 'decay'])
    parser.add_argument('--regularize_const', help='Regularization constant',
                        type=float, default=1e-4)
    parser.add_argument('--w_init', help='method used for initializing W',
                        type=str, default='var_scale', choices=['var_scale', 'xavier'])
    parser.add_argument('--data_format', help='data format',
                        type=str, default='channels_first', choices=['channels_first', 'channels_last'])
    parser.add_argument('--use_bias', help='Whether convolutions should use bias',
                        default=False, action='store_true')

    ## Special options to force input as uint8 and do mean/std process in graph in order to save memory
    # during cpu - gpu communication
    parser.add_argument('--input_type', help='Type for input, uint8 for certain dataset to speed up',
                        type=str, default='float32', choices=['float32', 'uint8'])
    parser.add_argument('--do_mean_std_gpu_process',
                        help='Whether use args.mean args.std to process in graph',
                        default=False, action='store_true')
    return parser, depth_group


################################################
# ResNet
################################################
def parser_add_resnet_arguments(parser):
    parser, depth_group = parser_add_common_arguments(parser)
    depth_group.add_argument('-d', '--depth',
                            help='depth of the network in number of conv',
                            type=int)

    parser.add_argument('--resnet_version', help='Version of resnet to use',
                        default='resnet', choices=['resnet', 'resnext'])
    return parser

class AnytimeNetwork(ModelDesc):
    def __init__(self, input_size, args):
        super(AnytimeNetwork, self).__init__()
        self.options = args

        self.data_format = args.data_format
        self.ch_dim = 1 if self.data_format == 'channels_first' else 3
        self.h_dim = 1 + int(self.data_format == 'channels_first')
        self.w_dim = self.h_dim + 1

        self.input_size = input_size
        self.network_config = compute_cfg(self.options)
        self.total_units = sum(self.network_config.n_units_per_block)

        # Warn user if they are using imagenet but doesn't have the right channel
        self.init_channel = args.init_channel
        self.n_blocks = len(self.network_config.n_units_per_block)
        self.cumsum_blocks = np.cumsum(self.network_config.n_units_per_block)
        self.width = args.width
        self.num_classes = self.options.num_classes
        self.alter_label = self.options.alter_label
        self.alter_label_activate_frac = self.options.alter_label_activate_frac
        self.alter_loss_w = self.options.alter_loss_w

        self.options.ls_method = self.options.samloss
        if self.options.ls_method == ADALOSS_LS_METHOD:
            self.options.is_select_arr = True
            self.options.sum_rand_ratio = 0.0
            assert self.options.func_type != FUNC_TYPE_OPT

        self.weights = anytime_loss.loss_weights(self.total_units, self.options,
            cfg=self.network_config.n_units_per_block)
        self.weights_sum = np.sum(self.weights)
        self.ls_K = np.sum(np.asarray(self.weights) > 0)
        logger.info('weights: {}'.format(self.weights))

        # special names and conditions
        self.select_idx_name = "select_idx"

        # (UGLY) due to the history of development. 1,...,5 requires rewards
        self.options.require_rewards = self.options.samloss < 6 and \
            self.options.samloss > 0

        if self.options.func_type == FUNC_TYPE_OPT \
            and self.options.ls_method != NO_AANN_METHOD:
            # special case: if we are computing optimal, don't do AANN
            logger.warn("Computing optimal requires not running AANN."\
                +" Setting samloss to be {}".format(NO_AANN_METHOD))
            self.options.ls_method = NO_AANN_METHOD
            self.options.samloss = NO_AANN_METHOD

        self.input_type = tf.float32 if self.options.input_type == 'float32' else tf.uint8
        if self.options.do_mean_std_gpu_process:
            if not hasattr(self.options, 'mean'):
                raise Exception('gpu_graph expects mean but it is not in the options')
            if not hasattr(self.options, 'std'):
                raise Exception('gpu_graph expects std, but it is not in the options')

        logger.info('the final options: {}'.format(self.options))


    def inputs(self):
        return [tf.TensorSpec(
                    [None, self.input_size, self.input_size, 3], 
                    self.input_type, 'input'),
                tf.TensorSpec(
                    [None], tf.int32, 'label')]

    def compute_scope_basename(self, layer_idx):
        return "layer{:03d}".format(layer_idx)

    def prediction_scope(self, scope_base):
        return scope_base + '.0.pred'


    ## Given the index (0based) of a block, return its scale.
    # This is overloaded by FCDenseNets.
    # Large scale_idx means smaller feature map.
    def bi_to_scale_idx(self, bi):
        return bi


    ###
    #   NOTE
    #   Important annoyance alert:
    #   Since this method is called typically before the build_graph is called,
    #   we cannot know the var/tensor names dynamically during cb construction.
    #   Hence it's up to implementation to make sure the right names are used.
    #
    #   To fix this, we need build_graph to know it's in test mode and construct
    #   right initialization, cbs, and surpress certain cbs/summarys.
    #   NOTE
    def compute_classification_callbacks(self):
        """
        """
        vcs = []
        total_units = self.total_units
        unit_idx = -1
        layer_idx=-1
        for n_units in self.network_config.n_units_per_block:
            for k in range(n_units):
                layer_idx += 1
                unit_idx += 1
                weight = self.weights[unit_idx]
                if weight > 0:
                    scope_name = self.compute_scope_basename(layer_idx)
                    scope_name = self.prediction_scope(scope_name) + '/'
                    vcs.append(ClassificationError(\
                        wrong_tensor_name=scope_name+'wrong-top1:0',
                        summary_name=scope_name+'val_err'))
                    vcs.append(ClassificationError(\
                        wrong_tensor_name=scope_name+'wrong-top5:0',
                        summary_name=scope_name+'val-err5'))
        return vcs

    def compute_loss_select_callbacks(self):
        logger.info("AANN samples with method {}".format(self.options.ls_method))
        if self.options.ls_method > 0 and not self.options.is_select_arr:
            reward_names = [ 'tower0/reward_{:02d}:0'.format(i) for i in range(self.ls_K)]
            select_idx_name = '{}:0'.format(self.select_idx_name)
            if self.options.ls_method == 3:
                online_learn_cb = FixedDistributionCPU(self.ls_K, select_idx_name, None)
            elif self.options.ls_method == 6:
                online_learn_cb = FixedDistributionCPU(self.ls_K, select_idx_name,
                    self.weights[self.weights>0])
            elif self.options.ls_method == 7:
                # custom schedule. select_idx will be initiated for use.
                # set the cb to be None to force use to give
                # a custom schedule/selector cb
                online_learn_cb = None
            else:
                gamma = self.options.exp_gamma
                if self.options.ls_method == 1:
                    online_learn_func = Exp3CPU
                    gamma = 1.0
                elif self.options.ls_method == 2:
                    online_learn_func = Exp3CPU
                elif self.options.ls_method == 4:
                    online_learn_func = RWMCPU
                elif self.options.ls_method == 5:
                    online_learn_func = ThompsonSamplingCPU
                online_learn_cb = online_learn_func(self.ls_K, gamma,
                    select_idx_name, reward_names)
            online_learn_cbs = [ online_learn_cb ]

        elif self.options.ls_method > 0:
            # implied self.options.is_select_arr == True
            loss_names = ['tower0/anytime_cost_{:02d}'.format(i) \
                    for i in range(self.ls_K)]
            select_idx_name = '{}:0'.format(self.select_idx_name)
            gamma = self.options.exp_gamma
            online_learn_cb = AdaptiveLossWeight(self.ls_K,
                    select_idx_name, loss_names,
                    gamma=self.options.adaloss_gamma,
                    update_per=self.options.adaloss_update_per,
                    momentum=self.options.adaloss_momentum,
                    final_extra=self.options.adaloss_final_extra)
            online_learn_cbs = [online_learn_cb]

        else:
            online_learn_cbs = []
        return online_learn_cbs

    def _compute_init_l_feats(self, image):
        l_feats = []
        with tf.variable_scope('init_conv0') as scope:
            if self.network_config.s_type == 'basic':
                l = Conv2D('conv0', image, self.init_channel, 3)
            else:
                assert self.network_config.s_type == 'imagenet'
                l = (LinearWrap(image)
                    .Conv2D('conv0', self.init_channel, 7, strides=2, activation=BNReLU)
                    .MaxPooling('pool0', 3, strides=2, padding='same')())
            l_feats.append(l)
        return l_feats

    def _compute_ll_feats(self, image):
        raise Exception("Invoked the base AnytimeNetwork. Use a specific one instead")

    def _compute_prediction_and_loss(self, l, label_obj, unit_idx):
        """
            l: feat_map of self.data_format
            label_obj: target to determine the loss
            unit_idx : the feature computation unit index.
        """
        l = GlobalAvgPooling('gap', l)
        avg_cost = 0
        avg_logits = 0
        variables = []
        for repeat_idx in list(range(self.options.num_predictor_copies)):
            postfix = '_' + str(repeat_idx) if repeat_idx > 0 else ""
            logits = FullyConnected('linear'+postfix, l, self.num_classes, activation=tf.identity)
            variables.append(logits.variables.W)
            variables.append(logits.variables.b)
            if self.options.high_temperature > 1.0:
                logits /= self.options.high_temperature

            ## local cost/error_rate
            label = label_obj[0]
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits, labels=label)
            cost = tf.reduce_mean(cost, name='cross_entropy_loss'+postfix)
            add_moving_summary(cost)
            
            def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
                return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)),
                   tf.float32, name=name)
            
            wrong = prediction_incorrect(logits, label, 1, name='wrong-top1'+postfix)
            add_moving_summary(tf.reduce_mean(wrong, name='train_error'+postfix))

            wrong5 = prediction_incorrect(logits, label, 5, name='wrong-top5'+postfix)
            add_moving_summary(tf.reduce_mean(wrong5, name='train-error-top5'+postfix))

            avg_cost += cost
            avg_logits += logits

        #return logits, cost
        avg_cost /= self.options.num_predictor_copies
        avg_logits /= self.options.num_predictor_copies
        return avg_cost, variables


    def _parse_inputs(self, inputs):
        """
            Parse the inputs so that it's split into image, followed by a "label" object.
            Note that the label object may contain multiple labels, such as coarse labels,
            and soft labels.

            The first returned variable is always the image as it was inputted
        """
        image = inputs[0]
        label = inputs[1:]
        return image, label

    def build_graph(self, *inputs):
        logger.info("sampling loss with method {}".format(self.options.ls_method))
        select_idx = self._get_select_idx()

        with argscope([Conv2D, Deconv2D, GroupedConv2D, AvgPooling, MaxPooling, BatchNorm, GlobalAvgPooling, ResizeImages],
                      data_format=self.data_format), \
            argscope([Conv2D, Deconv2D, GroupedConv2D], activation=tf.identity, use_bias=self.options.use_bias), \
            argscope([Conv2D, GroupedConv2D]), \
            argscope([BatchNorm], decay=self.options.batch_norm_decay):

            image, label = self._parse_inputs(inputs)

            # Common GPU side preprocess (uint8 -> float32), mean std, NCHW.
            if self.input_type == tf.uint8:
                image = tf.cast(image, tf.float32) * (1.0 / 255)
            if self.options.do_mean_std_gpu_process:
                if self.options.mean is not None:
                    image = image - tf.constant(self.options.mean, dtype=tf.float32)
                if self.options.std is not None:
                    image = image / tf.constant(self.options.std, dtype=tf.float32)
            if self.data_format == 'channels_first':
                image = tf.transpose(image, [0,3,1,2])

            self.dynamic_batch_size = tf.identity(tf.shape(image)[0], name='dynamic_batch_size')
            tf.reduce_mean(image, name='dummy_image_mean')
            ll_feats = self._compute_ll_feats(image)

            if self.options.stop_gradient:
                # NOTE:
                # Do not regularize for stop-gradient case, because
                # stop-grad requires cycling lr, and switching training targets
                wd_w = 0
            elif self.options.regularize_coef == 'const':
                wd_w = self.options.regularize_const
            elif self.options.regularize_coef == 'decay':
                wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                                  480000, 0.2, True)

            wd_cost = 0.0
            total_cost = 0.0
            unit_idx = -1
            anytime_idx = -1
            last_cost = None
            max_reward = 0.0
            online_learn_rewards = []
            for layer_idx, l_feats in enumerate(ll_feats):
                scope_name = self.compute_scope_basename(layer_idx)
                pred_scope = self.prediction_scope(scope_name)
                unit_idx += 1
                cost_weight = self.weights[unit_idx]
                if cost_weight == 0:
                    continue
                ## cost_weight is implied to be >0
                anytime_idx += 1
                with tf.variable_scope(pred_scope) as scope:
                    ## compute logit using features from layer layer_idx
                    l = tf.nn.relu(l_feats[0])
                    ch_in = l.get_shape().as_list()[self.ch_dim]
                    if self.options.prediction_feature == '1x1':
                        ch_out = int(self.options.prediction_feature_ch_out_rate * ch_in)
                        l = Conv2D('conv1x1', l, ch_out, 1)
                        l = BNReLU('bnrelu1x1', l)
                    elif self.options.prediction_feature == 'msdense':
                        if self.network_config.s_type == 'basic':
                            ch_inter = 128
                        else:
                            ch_inter = ch_in
                        l = Conv2D('conv1x1_0', l, ch_inter, 3, strides=2)
                        l = BNReLU('bnrelu1x1_0', l)
                        l = Conv2D('conv1x1_1', l, ch_inter, 3, strides=2)
                        l = BNReLU('bnrelu1x1_1', l)
                    elif self.options.prediction_feature == 'rescale':
                        bi = bisect.bisect_right(self.cumsum_blocks, unit_idx)
                        if bi+1 < len(self.cumsum_blocks):
                            l = ResizeImages('bilin_resize', l, [7,7], align_corners=False)
                            shape_list = [None,7,7,ch_in] if self.data_format == 'channels_last' \
                                else [None,ch_in,7,7]
                            l.set_shape(tf.TensorShape(shape_list))
                        l = Conv2D('conv1x1', l, ch_in, 1, activation=BNReLU)
                            #l = Conv2D('conv3x3_0', l, ch_in, 3, strides=2, activation=BNReLU)
                            #l = Conv2D('conv3x3_1', l, ch_in, 3, strides=2, activation=BNReLU)
                    elif self.options.prediction_feature == 'bn':
                        l = BatchNorm('bn', l)

                    cost, variables = self._compute_prediction_and_loss(l, label, unit_idx)
                #end with scope

                ## cost at each anytime prediction for learning the weights
                if self.options.adaloss_order == 0:
                    anytime_cost_i = tf.identity(cost,
                        name='anytime_cost_{:02d}'.format(anytime_idx))
                elif self.options.adaloss_order == 1:
                    logger.info("The gradient at predictor {:02d} is from layer {:03d}".format(anytime_idx, layer_idx))
                    grad = tf.gradients(cost, tf.trainable_variables())
                    grad_norm = tf.add_n([tf.nn.l2_loss(g) for g in grad if g is not None])
                    grad_dims = tf.add_n([tf.reshape(g, [-1]).get_shape().as_list()[0] for g in grad if g is not None])
                    anytime_cost_i = tf.identity(grad_norm / tf.cast(grad_dims, tf.float32),
                        name='anytime_cost_{:02d}'.format(anytime_idx))

                ## Compute the contribution of the cost to total cost
                # Additional weight for unit_idx.
                add_weight = 0
                if select_idx is not None and not self.options.is_select_arr:
                    add_weight = tf.cond(tf.equal(anytime_idx,
                                                  select_idx),
                        lambda: tf.constant(self.weights_sum,
                                            dtype=tf.float32),
                        lambda: tf.constant(0, dtype=tf.float32))

                elif select_idx is not None:
                    # implied is_select_arr == True
                    add_weight = select_idx[anytime_idx]

                if self.options.sum_rand_ratio > 0:
                    total_cost += (cost_weight + add_weight / \
                        self.options.sum_rand_ratio) * cost
                else:
                    total_cost += add_weight * cost

                ## Regularize weights from FC layers.
                logger.info("variables are {}".format(variables))
                for var in variables:
                    wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var)

                ################# (For 2017 submissions; Depricated) ###############
                ## Compute reward for loss selecters.

                # Compute relative loss improvement as rewards
                # note the rewards are outside varscopes.
                if self.options.require_rewards:
                    if not last_cost is None:
                        reward = 1.0 - cost / last_cost
                        max_reward = tf.maximum(reward, max_reward)
                        online_learn_rewards.append(tf.multiply(reward, 1.0,
                            name='reward_{:02d}'.format(anytime_idx-1)))
                    if anytime_idx == self.ls_K - 1:
                        reward = max_reward * self.options.last_reward_rate
                        online_learn_rewards.append(tf.multiply(reward, 1.0,
                            name='reward_{:02d}'.format(anytime_idx)))
                        #cost = tf.Print(cost, online_learn_rewards)
                    last_cost = cost
                ################# (End Depricated) ###############

                #end if compute_rewards
                #end (implied) if cost_weight > 0
            #endfor each layer
        #end argscope

        # weight decay on all W on conv layers for regularization
        wd_cost = tf.add(wd_cost, wd_w * regularize_cost('.*conv.*/W', tf.nn.l2_loss))
        wd_cost = tf.identity(wd_cost, name='wd_cost')
        total_cost = tf.identity(total_cost, name='sum_losses')
        add_moving_summary(total_cost, wd_cost)
        self.cost = tf.add_n([total_cost, wd_cost], name='cost') # specify training loss
        # monitor W # Too expensive in disk space :-/
        #add_param_summary(('.*/W', ['histogram']))
        return self.cost

    def optimizer(self):
        assert self.options.init_lr > 0, self.options.init_lr
        lr = tf.get_variable(
            'learning_rate', initializer=float(self.options.init_lr),
            trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = None
        if hasattr(self.options, 'optimizer'):
            if self.options.optimizer == 'rmsprop':
                logger.info('RMSPropOptimizer')
                opt = tf.train.RMSPropOptimizer(lr)
        if opt is None:
            logger.info('No optimizer was specified, using default MomentumOptimizer')
            opt = tf.train.MomentumOptimizer(lr, self.options.sgd_moment)
        return opt


    def _get_select_idx(self):
        select_idx = None

        if self.options.ls_method > 0 and not self.options.is_select_arr:
            init_idx = self.options.init_select_idx
            if init_idx is None:
                init_idx = self.ls_K - 1
            elif init_idx < 0:
                init_idx = self.ls_K + init_idx
            assert (init_idx >=0 and init_idx < self.ls_K), init_idx
            select_idx = tf.get_variable(self.select_idx_name, (), tf.int32,
                initializer=tf.constant_initializer(init_idx), trainable=False)
            tf.summary.scalar(self.select_idx_name, select_idx)
            for i in range(self.ls_K):
                weight_i = tf.cast(tf.equal(select_idx, i), tf.float32,
                    name='weight_{:02d}'.format(i))
                #add_moving_summary(weight_i)

        elif self.options.ls_method > 0:
            # implied self.options.is_select_arr == True
            select_idx = tf.get_variable(self.select_idx_name, (self.ls_K,),
                    tf.float32,
                    initializer=tf.constant_initializer([1.0]*self.ls_K),
                    trainable=False)
            for i in range(self.ls_K):
                weight_i = tf.identity(select_idx[i],
                    'weight_{:02d}'.format(i))
                add_moving_summary(weight_i)

        return select_idx

class AnytimeResNet(AnytimeNetwork):
    def __init__(self, input_size, args):
        super(AnytimeResNet, self).__init__(input_size, args)

    def residual_basic(self, name, l_feats, increase_dim=False):
        """
        Basic residual function for WANN: for index w,
        the input feat is the concat of all featus upto and including
        l_feats[w]. The output should have the same dimension
        as l_feats[w] for each w, if increase_dim is False

        Residual unit contains two 3x3 conv. The input is added
        to the final result of the conv path.
        The identity path has no bn/relu.
        The conv path has preact (bn); in between the two conv
        there is a bn_relu; the final conv is followed by bn.

        Note the only relu is in between convs. (following pyramidial net)

        When dim increases, the first conv has stride==2, and doubles
        the dimension. The final addition pads the new channels with zeros.

        """
        shape = l_feats[0].get_shape().as_list()
        in_channel = shape[self.ch_dim]

        if increase_dim:
            out_channel = in_channel * 2
            stride1 = 2
        else:
            out_channel = in_channel
            stride1 = 1

        l_mid_feats = []
        with tf.variable_scope(name+'.0.mid') as scope:
            l = BatchNorm('bn0', l_feats[0])
            l = tf.nn.relu(l)
            l = Conv2D('conv1', l, out_channel, 3, strides=stride1)
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l_mid_feats.append(l)

        l_end_feats = []
        with tf.variable_scope(name+'.0.end') as scope:
            l = l_mid_feats[0]
            ef = Conv2D('conv2', l, out_channel, 3)
            # The second conv need to be BN before addition.
            ef = BatchNorm('bn2', ef)
            l = l_feats[0]
            if increase_dim:
                l = AvgPooling('pool', l, 2)
                pad_paddings = [[0,0], [0,0], [0,0], [0,0]]
                pad_paddings[self.ch_dim] = [ in_channel//2, in_channel//2 ]
                l = tf.pad(l, pad_paddings)
            ef += l
            l_end_feats.append(ef)
        return l_end_feats

    def residual_bottleneck(self, name, l_feats, ch_in_to_ch_base=4):
        """
        Bottleneck resnet unit for WANN. Input of index w, is
        the concat of l_feats[0], ..., l_feats[w].

        The input to output has two paths. Identity paths has no activation.
        If the dimensions of in/output mismatch, input is converted to
        output dim via 1x1 conv with strides.

        The input channel of each l_feat is ch_in;
        the base channel is ch_base = ch_in // ch_in_to_ch_base.
        The conv paths contains three conv. 1x1, 3x3, 1x1.
        The first two convs outputs have channel(depth) of ch_base.
        The last conv has ch_base*4 output channels

        Within the same block, ch_in_to_ch_base is 4.
        The first res unit has ch_in_to_ch_base 1.
        The first res unit of other blocks has ch_in_to_ch_base 2.

        ch_in_to_ch_base == 2 also triggers downsampling with stride 2 at 3x3 conv

        """
        assert ch_in_to_ch_base in [1,2,4], ch_in_to_ch_base
        ch_in = l_feats[0].get_shape().as_list()[self.ch_dim]
        ch_base = ch_in // ch_in_to_ch_base

        stride=1
        if ch_in_to_ch_base == 2:
            # the first unit of block 2,3,4,... (1based)
            stride = 2

        l_new_feats = []
        with tf.variable_scope('{}.0.0'.format(name)) as scope:
            l = BatchNorm('bn0', l_feats[0])
            # according to pyramidal net, do not use relu here
            l = tf.nn.relu(l)
            l = (LinearWrap(l)
                .Conv2D('conv1x1_0', ch_base, 1, activation=BNReLU)
                .Conv2D('conv3x3_1', ch_base, 3, strides=stride, activation=BNReLU)
                .Conv2D('conv1x1_2', ch_base*4, 1)())
            l = BatchNorm('bn_3', l)

            shortcut = l_feats[0]
            if ch_in_to_ch_base < 4:
                shortcut = Conv2D('conv_short', shortcut, ch_base*4, 1, strides=stride)
                shortcut = BatchNorm('bn_short', shortcut)
            l = l + shortcut
            l_new_feats.append(l)
        return l_new_feats

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)

        ll_feats = []
        unit_idx = -1
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            for k in range(n_units):
                layer_idx += 1
                scope_name = self.compute_scope_basename(layer_idx)
                if self.network_config.b_type == 'basic':
                    l_feats = self.residual_basic(scope_name, l_feats, \
                        increase_dim=(k==0 and bi > 0))
                else:
                    assert self.network_config.b_type == 'bottleneck'
                    ch_in_to_ch_base = 4
                    if k == 0:
                        ch_in_to_ch_base = 2
                        if bi == 0:
                            ch_in_to_ch_base = 1
                    l_feats = self.residual_bottleneck(scope_name, l_feats, \
                        ch_in_to_ch_base)
                ll_feats.append(l_feats)

                # In case that we need to stop gradients
                is_last_row = bi==self.n_blocks-1 and k==n_units-1
                if self.options.stop_gradient and not is_last_row:
                    l_new_feats = []
                    for fi, f in enumerate(l_feats):
                        unit_idx +=1
                        if self.weights[unit_idx] > 0:
                            f = (1-self.options.sg_gamma)*tf.stop_gradient(f) \
                               + self.options.sg_gamma*f
                            logger.info("stop gradient after unit {}".format(unit_idx))
                        l_new_feats.append(f)
                    l_feats = l_new_feats
            # end for each k in n_units
        #end for each block
        return ll_feats


class AnytimeResNeXt(AnytimeResNet):

    def __init__(self, input_size, args):
        super(AnytimeResNeXt, self).__init__(input_size, args)
        # ResNeXt always uses bottleneck structure
        self.num_paths = 32


    def compute_ch_per_path(self, path, ch_base):
        # ratio of ch_base to ch_per_path
        # ratio table: (1,1), (2, 1.60), (4, 2.67), (8, 4.67), (32, 16.11)
        ratio = (8 * path + np.sqrt(64 * path**2 + 612 * path)) / 34.
        ch_per_path = int(np.ceil(ch_base / ratio))
        return ch_per_path


    def residual_bottleneck(self, name, l_feats, ch_in_to_ch_base=4):
        assert ch_in_to_ch_base in [1,2,4], ch_in_to_ch_base
        ch_in = l_feats[0].get_shape().as_list()[self.ch_dim]
        ch_base = ch_in // ch_in_to_ch_base
        ch_per_path = self.compute_ch_per_path(self.num_paths, ch_base)

        stride=1
        if ch_in_to_ch_base == 2:
            # the first unit of block 2,3,4,... (1based)
            stride = 2

        l_new_feats = []
        with tf.variable_scope('{}.0.0'.format(name)) as scope:
            l = BatchNorm('bn0', l_feats[0])
            l = tf.nn.relu(l)
            l = (LinearWrap(l)
                .Conv2D('conv1x1_0', self.num_paths * ch_per_path, 1, activation=BNReLU)
                .Conv2D('conv3x3_1', self.num_paths * ch_per_path, 3, strides=stride, \
                    activation=BNReLU, split=self.num_paths)
                .Conv2D('conv1x1_2', ch_base*4, 1)())
            l = BatchNorm('bn_3', l)

            shortcut = l_feats[0]
            if ch_in_to_ch_base < 4:
                shortcut = Conv2D('conv_short', shortcut, ch_base*4, 1, strides=stride)
                shortcut = BatchNorm('bn_short', shortcut)
            l = l + shortcut
            l_new_feats.append(l)
        return l_new_feats



################################################
# Dense Net (Log Dense)
################################################
def parser_add_densenet_arguments(parser):
    parser, depth_group = parser_add_common_arguments(parser)
    depth_group.add_argument('--densenet_depth',
                             help='depth of densenet for predefined networks',
                             type=int)
    parser.add_argument('--densenet_version', help='specify the version of densenet to use',
                        type=str, default='atv1', choices=['atv1', 'atv2', 'dense', 'loglog', 'c2f'])
    parser.add_argument('-g', '--growth_rate', help='growth rate k for log dense',
                        type=int, default=16)
    parser.add_argument('--bottleneck_width', help='multiplier of growth for width of bottleneck',
                        type=float, default=4.0)
    parser.add_argument('--growth_rate_multiplier',
                        help='a constant to multiply growth_rate by at pooling',
                        type=int, default=1, choices=[1,2])
    parser.add_argument('--use_init_ch',
                        help='whether to use specified init channel argument, '\
                            +' useful for networks that has specific init_ch based on'\
                            +' other metrics such as densenet',
                        default=False, action='store_true')
    parser.add_argument('--dense_select_method', help='densenet previous feature selection choice',
                        type=int, default=0)
    parser.add_argument('--early_connect_type', help='Type of forced early conneciton_types',
                        type=int, default=0)
    parser.add_argument('--log_dense_coef', help='The constant multiplier of log(depth) to connect',
                        type=np.float32, default=1)
    parser.add_argument('--log_dense_base', help='base of log',
                        type=np.float32, default=2)
    parser.add_argument('--transition_batch_size',
                        help='number of layers to transit together per conv; ' +\
                             '-1 means all previous layers transition together using 1x1 conv',
                        type=int, default=1)
    parser.add_argument('--use_transition_mask',
                        help='When transition together, whether use W_mask to force indepence',
                        default=False, action='store_true')
    parser.add_argument('--pre_activate', help='whether BNReLU pre conv or after',
                        type=int, default=1, choices=[0, 1])
    parser.add_argument('--dropout_kp', help='Dropout probability',
                        type=np.float32, default=1.0)
    parser.add_argument('--reduction_ratio', help='reduction ratio at transitions',
                        type=np.float32, default=1.0)
    parser.add_argument('--loglog_growth_multiplier', help='Loglog recursion depth 0 growth rate multiplier',
                        type=np.float32, default=1.0)
    return parser, depth_group


class DenseNet(AnytimeNetwork):
    """
        This class is for reproducing densenet results.
    """
    def __init__(self, input_size, args):
        super(DenseNet, self).__init__(input_size, args)
        self.reduction_ratio = self.options.reduction_ratio
        self.growth_rate = self.options.growth_rate
        self.bottleneck_width = self.options.bottleneck_width
        self.dropout_kp = self.options.dropout_kp

        if not self.options.use_init_ch:
            default_ch = self.growth_rate * 2
            if self.init_channel != default_ch:
                self.init_channel = default_ch
                logger.info("Densenet sets the init_channel to be " \
                    + "2*growth_rate by default. " \
                    + "I'm setting this automatically!")


    def compute_block(self, pmls, layer_idx, n_units, growth):
        pml = pmls[-1]
        if layer_idx > -1:
            with tf.variable_scope('transit_after_{}'.format(layer_idx)) as scope:
                ch_in = pml.get_shape().as_list()[self.ch_dim]
                ch_out = int(ch_in * self.reduction_ratio)
                pml = BNReLU('trans_bnrelu', pml)
                pml = Conv2D('conv1x1', pml, ch_out, 1)
                pml = Dropout('dropout', pml, keep_prob=self.dropout_kp)
                pml = AvgPooling('pool', pml, 2, padding='same')

        for k in range(n_units):
            layer_idx +=1
            scope_name = self.compute_scope_basename(layer_idx)
            with tf.variable_scope(scope_name) as scope:
                l = pml
                l = BNReLU('pre_bnrelu', l)
                if self.network_config.b_type == 'bottleneck':
                    bnw = int(self.bottleneck_width * growth)
                    l = Conv2D('conv1x1', l, bnw, 1)
                    l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                    l = BNReLU('bottleneck_bnrelu', l)
                l = Conv2D('conv3x3', l, growth, 3)
                l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                pml = tf.concat([pml, l], self.ch_dim, name='concat')
                pmls.append(pml)
        return pmls

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        pmls = [l_feats[0]]
        growth = self.growth_rate
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            pmls = self.compute_block(pmls, layer_idx, n_units, growth)
            layer_idx += n_units

        pmls = pmls[1:]
        ll_feats = [ [ BNReLU('bnrelu_{}'.format(li), ml) ] if self.weights[li] > 0 else [None]
            for li, ml in enumerate(pmls) ]
        return ll_feats


class AnytimeLogDenseNetV1(AnytimeNetwork):
    def __init__(self, input_size, args):
        super(AnytimeLogDenseNetV1, self).__init__(input_size, args)
        self.dense_select_method = self.options.dense_select_method
        self.log_dense_coef = self.options.log_dense_coef
        self.log_dense_base = self.options.log_dense_base
        self.reduction_ratio = self.options.reduction_ratio
        self.growth_rate = self.options.growth_rate
        self.growth_rate_multiplier = self.options.growth_rate_multiplier
        self.bottleneck_width = self.options.bottleneck_width
        self.early_connect_type = self.options.early_connect_type

        ## deprecated. don't use
        self.transition_batch_size = self.options.transition_batch_size
        self.use_transition_mask = self.options.use_transition_mask
        self.dropout_kp = self.options.dropout_kp

        if not self.options.use_init_ch:
            default_ch = self.growth_rate * 2
            if self.init_channel != default_ch:
                self.init_channel = default_ch
                logger.info("Densenet sets the init_channel to be " \
                    + "2*growth_rate by default. " \
                    + "I'm setting this automatically!")

        if self.options.func_type == FUNC_TYPE_ANN \
            and self.options.ls_method != BEST_AANN_METHOD:
            logger.warn("Densenet prefers using AANN instead of other methods."\
                +"Changing samloss to be {}".format(BEST_AANN_METHOD))
            self.options.ls_method = BEST_AANN_METHOD
            self.options.samloss = BEST_AANN_METHOD

        ## pre-compute connection; without connection augmentations.
        self._connections = None
        # For each ui, an input list; this contains forced connections
        self.connections = None
        # Used for determine whether a layer needs transition
        # exists for each pls, including the init conv
        # scale 0 means input image; large scale (idx) means smaller feature map
        self.l_max_scale = None
        self.l_min_scale = None

        ## NOTE on how the connections are formed and used
        #
        # 1. In the end, we use self.connections[ui] (0-based) to retrieve connections.
        #  To form self.connections[..], we call pre_compute_connections().
        #  This is typically done near the start of _compute_ll_feats
        #
        # 2. pre_compute_connections() first calls _pre_compute_connections(),
        #  which sets up _connections for certain complicated connection patterns.
        #  Then for each ui=0,..., we use dense_select_indices() to pre_compute
        #  the connections self.connections[ui], and update the required scales for
        #  each ui in self.l_min_scale and self.l_min_scale
        #
        # 3. dense_select_indices() has two cases, it either uses self._connections[ui]
        #  to retrive connections from _pre_compute_connections(), or call
        #  _dense_select_indices() to compute self._connections[ui] on the fly.
        #  Then self._connections[ui] is augmented with forced early connections,
        #  and are filtered by self.special_filter() to form self.connection[ui]


    ## Some networks need special grwoth rate for certain layers,
    # e.g., loglogdense
    def _compute_layer_growth(self, ui, growth):
        return growth


    ## Pre-compute connections that include all forced connects
    # such as _dense_select_early_connect
    def pre_compute_connections(self):
        self._pre_compute_connections()
        self.connections = []
        self.l_max_scale = [ 0 for _ in range(self.total_units + 1) ]
        self.l_min_scale = [ 0 for _ in range(self.total_units + 1) ]
        curr_bi = 0
        curr_scale = 0
        for ui in range(self.total_units):
            if self.cumsum_blocks[curr_bi] == ui:
                curr_bi += 1
                curr_scale = self.bi_to_scale_idx(curr_bi)
            self.connections.append(self.dense_select_indices(ui))
            for i in self.connections[-1]:
                self.l_max_scale[i] = max(self.l_max_scale[i], curr_scale)
                self.l_min_scale[i] = min(self.l_min_scale[i], curr_scale)
            self.l_min_scale[ui+1] = curr_scale
            self.l_max_scale[ui+1] = curr_scale


    ## Some connections methods requrie some recursion or complex
    # functions to set all the connection together first.
    # this will be set in
    #               self._connections.
    # it will then be used by pre_compute_connections and
    # dense_select_method to be augmented with forced connections
    # and special_filter
    def _pre_compute_connections(self):
        self._connections = None


    ## Check whether _connections is filled first;
    # if not then use _dense_select_indices to actually comput connections.
    def dense_select_indices(self, ui):
        if self._connections is not None:
            # the selections are precomputed.
            # get them for the stored list
            indices = self._connections[ui]
        else:
            # the selection is not defined yet.
            # compute dynamically right now
            indices = self._dense_select_indices(ui)
        indices = self._dense_select_early_connect(ui, indices)
        indices = filter(lambda x, ui=ui: x <=ui and x >=0 and \
            self.special_filter(ui, x), np.unique(indices))
        return indices


    ## whether pls[x] should be included for computing ui,
    # given that pls[x] is selected already by DSM, early forcement.
    # e.g., log-dense-v2 uses this to cut early connections to do block
    # compression
    def special_filter(self, ui, x):
        return True


    ## Multiple ways to force connections to early layers.
    # e.g.,
    # 0: connect to nothing;
    # 1: the end of the first block
    def _dense_select_early_connect(self, ui, indices):
        if self.early_connect_type == 0:
            # default 0 does nothing
            pass
        if self.early_connect_type % 2 ==  1:  #e.g., 1
            # force connection to end of first block
            indices.append(self.network_config.n_units_per_block[0])
        if (self.early_connect_type >> 1) % 2 == 1:  #e.g., 2
            # force connect to all of first block
            indices.extend(list(range(self.network_config.n_units_per_block[0]+1)))
        if (self.early_connect_type >> 2) % 2 == 1:  #e.g., 4
            # force connect to end of the first three blocks
            indices.extend(self.cumsum_blocks[:3])
        if (self.early_connect_type >> 3) % 2 == 1: # e.g., 8
            # force connect to end of all blocks
            indices.extend(self.cumsum_blocks)
        indices = filter(lambda x : x <=ui and x >=0, np.unique(indices))
        return indices


    def _dense_select_indices(self, ui):
        """
            Given ui, return the list of indices i's such that
            pls[i] contribute to forming layer[ui] i.e. pls[ui+1].
            For methods that can be computed directly using ui, and other
            args, use this method to do so.
            If the computation is too complex or it is easy to pre-compute
            all connections before hand, see loglogdense as an example, and
            use _pre_compute_connections.
            ui : unit_idx
        """
        if self.dense_select_method == 0:
            # log dense
            if not hasattr(self, 'exponential_diffs'):
                df = 1
                exponential_diffs = [0]
                while True:
                    df = df * self.log_dense_base
                    if df > self.total_units * self.log_dense_coef:
                        break
                    int_df = int((df-1) / self.log_dense_coef)
                    if int_df != exponential_diffs[-1]:
                        exponential_diffs.append(int_df)
                self.exponential_diffs = exponential_diffs
            diffs = self.exponential_diffs

        elif self.dense_select_method == 1:
            # all at the end with log(i)
            diffs = list(range(int(np.log2(ui + 1) \
                / np.log2(self.log_dense_base) * self.log_dense_coef) + 1))

        elif self.dense_select_method == 2:
            # all at the end with log(L)
            n_select = int((np.log2(self.total_units +1) \
                / np.log2(self.log_dense_base) + 1) * self.log_dense_coef)
            diffs = list(range(int(np.log2(self.total_units + 1)) + 1))

        elif self.dense_select_method == 3:
            # Evenly spaced connections (select log)
            n_select = int((np.log2(self.total_units +1) + 1) * self.log_dense_coef)
            delta = (ui+1.0) / n_select
            df = 0
            diffs = []
            for i in range(n_select):
                int_df = int(df)
                if len(diffs) == 0 or int_df != diffs[-1]:
                    diffs.append(int_df)
                df += delta

        elif self.dense_select_method == 4:
            # select all
            diffs = list(range(ui+1))

        elif self.dense_select_method == 5:
            # mini dense
            # For all x_i, x_i is close to x_L
            # No guarantees for BD(x_i, x_j) in general
            diffs = [0, 1]
            left = 0
            right = self.total_units + 1
            while right != ui+2: # seek ui+1
                mid = (left + right) // 2
                if ui+1 >= mid:
                    left = mid
                else:
                    right = mid
            df = right - (right + left) // 2 - 1
            if df > 1:
                diffs.append(df)

        elif self.dense_select_method == 6:
            # select at the end with 0.5 * i
            # BD(xi, xj) <= log(i-j)
            n_select = ui // 2 + 1
            diffs = list(range(n_select))

        elif self.dense_select_method == 7:
            # select from every other layer
            # starting from the previous layer (diff==0)
            # BD(xi, xj) <= 2
            diffs = list(range(0, ui+1, 2))

        elif self.dense_select_method == 8:
            # select at the end with 0.5 *i
            # also select the key hubs at 0.25 *i, 0.125*i, ...
            # BD(xi, xj) <= 2
            n_select = ui // 2 + 1
            diffs = list(range(n_select))
            pos = ui - n_select + 1
            while pos > 0:
                pos = (pos - 1) // 2
                diffs.append(ui - pos)

        indices = [ui - df  for df in diffs if ui - df >= 0 ]
        return indices


    def compute_block(self, pls, pmls, n_units, growth):
        """
            pls : previous layers. including the init_feat. Hence pls[i] is from
                layer i-1 for i > 0
            pmls : previous merged layers. (used for generate ll_feats)
            n_units : num units in a block

            return pls, pmpls (updated version of these)
        """
        unit_idx = len(pls) - 2 # unit idx of the last completed unit
        for _ in range(n_units):
            unit_idx += 1
            scope_name = self.compute_scope_basename(unit_idx)
            with tf.variable_scope(scope_name+'.feat'):
                sl_indices = self.connections[unit_idx]
                logger.info("unit_idx = {}, len past_feats = {}, selected_feats: {}".format(\
                    unit_idx, len(pls), sl_indices))

                ml = tf.concat([pls[sli] for sli in sl_indices], \
                               self.ch_dim, name='concat_feat')
                l = BNReLU('bnrelu_merged', ml)

                layer_growth = self._compute_layer_growth(unit_idx, growth)
                if self.network_config.b_type == 'bottleneck':
                    bottleneck_width = int(self.options.bottleneck_width * layer_growth)
                    l = Conv2D('conv1x1', l, bottleneck_width, 1, activation=BNReLU)
                    l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                    l = Conv2D('conv3x3', l, layer_growth, 3)
                else:
                    l = Conv2D('conv3x3', l, layer_growth, 3)
                l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                pls.append(l)

                # If the feature is used for prediction, store it.
                if self.weights[unit_idx] > 0:
                    pmls.append(tf.concat([ml, l], self.ch_dim, name='concat_pred'))
                else:
                    pmls.append(None)
        return pls, pmls


    ##
    # pls (list) : list of previous layers including the init conv
    # trans_idx (int) : curret block index bi, (the one just finished)
    def compute_transition(self, pls, trans_idx):
        new_pls = []
        for pli, pl in enumerate(pls):
            if self.l_max_scale[pli] <= trans_idx:
                new_pls.append(None)
                continue

            ch_in = pl.get_shape().as_list()[self.ch_dim]
            ch_out = int(ch_in * self.growth_rate_multiplier * self.reduction_ratio)

            with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)):
                pl = BNReLU('bnrelu_transit', pl)
                l = Conv2D('conv', pl, ch_out, 1)
                l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                l = AvgPooling('pool', l, 2, padding='same')
                new_pls.append(l)
        return new_pls


    def _compute_ll_feats(self, image):
        self.pre_compute_connections()
        l_feats = self._compute_init_l_feats(image)
        pls = [l_feats[0]]
        pmls = []
        growth = self.growth_rate
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            pls, pmls = self.compute_block(pls, pmls, n_units, growth)
            if bi != self.n_blocks - 1:
                growth *= self.growth_rate_multiplier
                pls = self.compute_transition(pls, bi)

        ll_feats = [ [ BNReLU('bnrelu_{}'.format(li), ml) ] if self.weights[li] > 0 else [None]
            for li, ml in enumerate(pmls) ]
        return ll_feats


class AnytimeLogDenseNetV2(AnytimeLogDenseNetV1):
    """
        This version of dense net will do block compression
        by compression a block into log L layers.
        Any future layer will use the entire log L layers,
        even in log-dense
    """
    def __init__(self, input_size, args):
        super(AnytimeLogDenseNetV2, self).__init__(input_size, args)


    def special_filter(self, ui, x):
        if ui == 0 or x == 0:
            return False
        bi = bisect.bisect_right(self.cumsum_blocks, ui)
        bi_x = bisect.bisect_right(self.cumsum_blocks, x-1)
        return bi == bi_x

    def compute_block(self, layer_idx, n_units, l_mls, bcml, growth):
        pls = []
        # offset is the first layer_idx that in this block.
        # layer_idx+1 now contains the last of the last block
        pli_offset = layer_idx + 2
        for k in range(n_units):
            layer_idx += 1
            scope_name = self.compute_scope_basename(layer_idx)
            with tf.variable_scope(scope_name):
                sl_indices = self.connections[layer_idx]
                logger.info("layer_idx = {}, len past_feats = {}, selected_feats: {}".format(\
                    layer_idx, len(pls), sl_indices))

                ml = tf.concat([bcml] + [pls[sli - pli_offset] for sli in sl_indices], \
                               self.ch_dim, name='concat_feat')
                l = BNReLU('bnrelu_merged', ml)
                if self.network_config.b_type == 'bottleneck':
                    bnw = int(self.bottleneck_width * growth)
                    l = Conv2D('conv1x1', l, bnw, 1, activation=BNReLU)
                    l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                    l = Conv2D('conv3x3', l, growth, 3)
                else:
                    l = Conv2D('conv3x3', l, growth, 3)
                l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                pls.append(l)

                if self.weights[layer_idx] > 0:
                    l = tf.concat([ml, l], self.ch_dim, name='concat_pred')
                    l_mls.append(l)
                else:
                    l_mls.append(None)
        return pls, l_mls


    def update_compressed_feature(self, layer_idx, ch_out, pls, bcml):
        """
            pls: new layers
            bcml : the compressed features for generating pls
        """
        with tf.variable_scope('transition_after_{}'.format(layer_idx)) as scope:
            l = tf.concat(pls, self.ch_dim, name='concat_new')
            ch_new = l.get_shape().as_list()[self.ch_dim]
            l = BNReLU('pre_bnrelu', l)
            l = Conv2D('conv1x1_new', l, min(ch_out, ch_new), 1)
            l = Dropout('dropout_new', l, keep_prob=self.dropout_kp)
            l = AvgPooling('pool_new', l, 2, padding='same')

            ch_old = bcml.get_shape().as_list()[self.ch_dim]
            bcml = BNReLU('pre_bnrelu_old', bcml)
            bcml = Conv2D('conv1x1_old', bcml, ch_old, 1)
            bcml = Dropout('dropout_old', bcml, keep_prob=self.dropout_kp)
            bcml = AvgPooling('pool_old', bcml, 2, padding='same')

            bcml = tf.concat([bcml, l], self.ch_dim, name='concat_all')
        return bcml


    def _compute_ll_feats(self, image):
        self.pre_compute_connections()
        l_feats = self._compute_init_l_feats(image)
        bcml = l_feats[0]
        l_mls = []
        growth = self.growth_rate
        layer_idx = -1
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            pls, l_mls = self.compute_block(layer_idx, n_units, l_mls, bcml, growth)
            layer_idx += n_units
            if bi != self.n_blocks - 1:
                ch_out = growth * (int(np.log2(self.total_units + 1)) + 1)
                bcml = self.update_compressed_feature(layer_idx, ch_out, pls, bcml)

        ll_feats = [ [ BNReLU('bnrelu_{}'.format(li), ml) ] if self.weights[li] > 0 else [None]
            for li, ml in enumerate(l_mls) ]
        assert len(ll_feats) == self.total_units
        return ll_feats


############################
# Log-Log dense
############################
class AnytimeLogLogDenseNet(AnytimeLogDenseNetV1):
    def __init__(self, input_size, args):
        super(AnytimeLogLogDenseNet, self).__init__(input_size, args)
        self._recursion_depths = None
        self.loglog_growth_multiplier = args.loglog_growth_multiplier


    def _compute_layer_growth(self, ui, growth):
        if self._recursion_depths[ui] == 0:
            return growth * self.loglog_growth_multiplier
        return growth


    ## pre compute connections for log-log dense as it is rather complicated
    # construct connection adjacency list for backprop
    # The pls starts with the initial conv which has index 0, so that it is 1-based.
    # The input ui is 0-based. Hence, e.g., ui (input) always connects to
    # pls[ui], which is the previous layer of input[ui].
    def _pre_compute_connections(self):
        # Everything is connected to the previous layer
        # l_adj[0] is a placeholder, so ignore that it connects to -1.
        l_adj = [ [i-1] for i in range(self.total_units+1) ]
        recursion_depths = [ -1 for i in range(self.total_units+1) ]

        ## padding connection offset to ensure at least bn_width connections are used.
        # i - offset is the input index
        padding_offsets = [1, 2, 4, 8, 16, 32]


        ## update l_adj connecitono on interval [a, b)
        def loglog_connect(a, b, depth, force_connect_locs=[]):
            if b-a <= 2:
                return None

            seg_len = b-a
            step_len = int(np.sqrt(b-a))
            key_indices = list(range(a + (b-1-a) % step_len, b, step_len))
            if len(force_connect_locs) > 0:
                for fc_key in force_connect_locs:
                    if not fc_key in key_indices:
                        key_indices.append(fc_key)
                key_indices = sorted(key_indices)
            if key_indices[-1] != b-1:
                key_indices.append(b-1)
            if key_indices[0] != a:
                key_indices.insert(0, a)

            # connection at the current recursion depth
            for ki, key in enumerate(key_indices):
                if recursion_depths[key] < 0:
                    recursion_depths[key] = depth
                if ki == 0:
                     continue
                for prev_key in key_indices[:ki]:
                    if not prev_key in l_adj[key]:
                        l_adj[key].append(prev_key)
                prev_key = key_indices[ki-1]
                for li in range(prev_key + 1, key):
                    if not prev_key in l_adj[li]:
                        l_adj[li].append(prev_key)
                loglog_connect(prev_key, key+1, depth+1)
            return None


        force_connect_locs = self.cumsum_blocks
        loglog_connect(0, self.total_units+1, 0, force_connect_locs)
        ## since the first conv is init conv that we don't count for pred.
        for i in range(self.total_units):
            l_adj[i+1] = filter(lambda x: x >= 0 and x <= i, np.unique(l_adj[i+1]))
            if len(l_adj[i+1]) < self.bottleneck_width:
                for offset in padding_offsets:
                    idx = i + 1 - offset
                    if idx < 0:
                        break
                    if not idx in l_adj[i+1]:
                        l_adj[i+1].append(idx)
                        if len(l_adj[i+1]) >= self.bottleneck_width:
                            break

        self._connections = l_adj[1:]
        self._recursion_depths = recursion_depths[1:]


###########################
# Multi-scale Dense-Network
###########################
def parser_add_msdensenet_arguments(parser):
    parser, depth_group = parser_add_common_arguments(parser)
    depth_group.add_argument('--msdensenet_depth',
                             help='depth of multiscale densenet', type=int)
    parser.add_argument('-g', '--growth_rate', help='growth rate at high resolution',
                        type=int, default=6)
    parser.add_argument('--bottleneck_width', help='multiplier of growth for width of bottleneck',
                        type=float, default=4.0)
    parser.add_argument('--num_scales', help='number of scales',
                        type=int, default=3)
    parser.add_argument('--reduction_ratio', help='reduction ratio between blocks',
                        type=float, default=0.5)
    parser.add_argument('--prune', help='Prune method min or max: prune minimum or maximum amount',
                        type=str, default='max', choices=['max'])
    parser.add_argument('--dropout_kp', help='Dropout keep probability',
                        type=float, default=1)


class AnytimeMultiScaleDenseNet(AnytimeNetwork):

    def __init__(self, input_size, args):
        super(AnytimeMultiScaleDenseNet, self).__init__(input_size, args)
        self.num_scales = self.options.num_scales
        self.growth_rate = self.options.growth_rate
        self.growth_rate_factor = [1,2,4,4]
        self.bottleneck_factor = [1,2,4,4]
        self.init_channel = self.growth_rate * 2
        self.reduction_ratio = self.options.reduction_ratio
        self.dropout_kp = args.dropout_kp


    def _compute_init_l_feats(self, image):
        l_feats = []
        for w in range(self.num_scales):
            scope_name = self.compute_scope_basename(0)
            with tf.variable_scope(scope_name+'.'+str(w)) as scope:
                ch_out = self.init_channel * self.growth_rate_factor[w]
                if w == 0:
                    if self.network_config.s_type == 'basic':
                        l = Conv2D('conv3x3', image, ch_out, 3, activation=BNReLU)
                    else:
                        assert self.network_config.s_type == 'imagenet'
                        l = (LinearWrap(image)
                            .Conv2D('conv7x7', ch_out, 7, strides=2, activation=BNReLU)
                            .MaxPooling('pool0', 3, strides=2, padding='same')())
                else:
                    l = Conv2D('conv3x3', l, ch_out, 3, strides=2, activation=BNReLU)
                l_feats.append(l)
        return l_feats

    def _compute_edge(self, l, ch_out, bnw, l_type='normal', dyn_hw=None, name=""):
        if self.network_config.b_type == 'bottleneck':
            bnw_ch = int(bnw * ch_out)
            l = Conv2D('conv1x1_'+name, l, bnw_ch, 1, activation=BNReLU)
        if l_type == 'up':
            assert dyn_hw is not None, dyn_hw
            l = ResizeImages('bilin_resize', l, dyn_hw)
            stride = 1
        else:
            if l_type == 'normal':
                stride = 1
            elif l_type == 'down':
                stride = 2
        l = Conv2D('conv3x3_'+name, l, ch_out, 3, strides=stride, activation=BNReLU)
        return l

    def _compute_block(self, bi, n_units, layer_idx, l_mf):
        ll_feats = []
        for k in range(n_units):
            layer_idx += 1
            scope_name = self.compute_scope_basename(layer_idx)
            s_start = bi
            l_feats = [None] * s_start
            for w in range(s_start, self.num_scales):
                with tf.variable_scope(scope_name+'.'+str(w)) as scope:
                    g = self.growth_rate_factor[w] * self.growth_rate
                    bnw = self.bottleneck_factor[w]
                    has_prev_scale = w > s_start or (w > 0 and k==0)
                    if not has_prev_scale:
                        l = self._compute_edge(l_mf[w], g, bnw, 'normal')
                    else:
                        l = self._compute_edge(l_mf[w], g/2, bnw, 'normal', name='e1')
                        bnw_prev = self.bottleneck_factor[w-1]
                        lp = self._compute_edge(l_mf[w-1], g - g/2, bnw_prev, 'down', name='e2')
                        l = tf.concat([l, lp], self.ch_dim, name='concat_ms')
                    l_feats.append(l)
            #end for w
            new_l_mf = [None] * self.num_scales
            for w in range(s_start, self.num_scales):
                with tf.variable_scope(scope_name+'.'+str(w)+'.merge') as scope:
                    new_l_mf[w] = tf.concat([l_mf[w], l_feats[w]],
                        self.ch_dim, name='merge_feats')
            ll_feats.append(new_l_mf)
            l_mf = new_l_mf
        #end for k in units
        return ll_feats


    def _compute_transition(self, ll_merged_feats, layer_idx):
        rr = self.reduction_ratio
        l_feats = []
        for w, l in enumerate(ll_merged_feats):
            if l is None:
                l_feats.append(None)
                continue

            with tf.variable_scope('transat_{:02d}_{:02d}'.format(layer_idx+1, w)):
                ch_in = l.get_shape().as_list()[self.ch_dim]
                ch_out = int(ch_in * rr)
                l = Conv2D('conv1x1', l, ch_out, 1, activation=BNReLU)
                l_feats.append(l)
        return l_feats


    def _compute_ll_feats(self, image):
        l_mf = self._compute_init_l_feats(image)
        ll_feats = [ l_mf ]
        layer_idx = 0
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            ll_block_feats = self._compute_block(bi, n_units-1, layer_idx, l_mf)
            layer_idx += n_units-1
            ll_feats.extend(ll_block_feats)
            l_mf = ll_block_feats[-1]
            if bi < self.n_blocks - 1:
                l_mf = self._compute_transition(l_mf, layer_idx)
                ll_feats.append(l_mf)
                layer_idx += 1

        ll_feats = [ [None if l_feats is None else l_feats[-1]] for l_feats in ll_feats ]
        return ll_feats
