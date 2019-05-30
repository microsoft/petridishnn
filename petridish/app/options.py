# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
from tensorpack.utils import logger
from petridish.info import (LayerTypes,
    LayerInfo, LayerInfoList, CellNetworkInfo, net_info_from_str)

__all__ = ['add_model_arguments', 'add_controller_arguments',
    'model_options_processing', 'options_to_str']

def add_app_arguments(parser):
    parser.add_argument(
        '--ds_name', help='name of dataset',
        type=str, default='ilsvrc')
    parser.add_argument(
        '--data_dir',
        help='ILSVRC dataset dir that contains the tf records directly',
        default=os.getenv('PT_DATA_DIR', 'data'))
    parser.add_argument(
        '--log_dir', help='log_dir for stdout',
        default=os.getenv('PT_OUTPUT_DIR', '/tmp'))
    parser.add_argument(
        '--model_dir', help='dir for saving models',
        default=os.getenv('PT_OUTPUT_DIR', '/tmp'))
    parser.add_argument(
        '--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument(
        '--do_validation', help='whether to use validation set',
        default=False, action='store_true')
    parser.add_argument(
        '--load', help='load model', default=None, type=str)
    return parser


def add_model_arguments(parser):
    ## Model construction params
    parser.add_argument(
        '--net_info_str', help='a list of layer types from LayerTypes',
        default=None, type=str)

    ## model params
    parser.add_argument(
        '--input_size', help='Side size of input images.',
        type=int, default=None)
    parser.add_argument(
        '-c', '--init_channel',
        help='number of channels at the first non stem/init cell/layer',
        type=int, default=16)
    parser.add_argument(
        '--stem_channel_rate', type=float, default=1,
        help='Multiplier for stem operation channels')
    parser.add_argument(
        '--s_type', help='starting conv type',
        type=str, default='basic',
        choices=['basic', 'imagenet', 'conv7', 'conv3'])
    parser.add_argument(
        '--b_type', help='block type',
        type=str, default='basic', choices=['basic', 'bottleneck'])
    parser.add_argument(
        '--prediction_feature',
        help='Type of feature processing for prediction',
        type=str, default='none',
        choices=['none', '1x1', 'msdense', 'bn', 'rescale'])

    ## Anytime prediction related.
    parser.add_argument(
        '--ls_method', '--samloss',
        help='Method to Sample losses to update. 0 for static. 100 for adaloss.',
        type=int, default=0)
    parser.add_argument(
        '--adaloss_gamma', help='Gamma for adaloss',
        type=np.float32, default=0.07)
    parser.add_argument(
        '--adaloss_momentum', help='Adaloss momentum',
        type=np.float32, default=0.9)
    parser.add_argument(
        '--adaloss_update_per',
        help='Adaloss update weights every number of iter',
        type=int, default=100)
    parser.add_argument(
        '--adaloss_final_extra',
        help='Adaloss up-weights the final loss',
        type=np.float32, default=0.0)

    ## misc: training params, data-set params, speed/memory params
    parser.add_argument(
        '--batch_size_per_gpu', help='Batch size for train/testing',
        type=int, default=None)

    parser.add_argument(
        '--batch_norm_decay', help='decay rate of batchnorms',
        type=np.float32, default=0.9)
    parser.add_argument(
        '--batch_norm_epsilon', help='margin for zero',
        type=np.float32, default=1e-5)
    parser.add_argument(
        '--num_classes', help='Number of classes',
        type=int, default=10)
    parser.add_argument(
        '--regularize_coef',
        help='How coefficient of regularization decay',
        type=str, default='const', choices=['const', 'decay'])
    parser.add_argument(
        '--regularize_const', help='Regularization constant for weight decay',
        type=float, default=1e-4)
    parser.add_argument(
        '--w_init', help='method used for initializing W',
        type=str, default='var_scale', choices=['var_scale', 'xavier'])
    parser.add_argument(
        '--data_format', help='data format',
        type=str, default='channels_first',
        choices=['channels_first', 'channels_last'])
    parser.add_argument(
        '--use_bias', help='Whether convolutions should use bias',
        default=False, action='store_true')
    parser.add_argument(
        '--training_type', default=None, type=str,
        help=('Preset training params. Each data-set has its own pre-definition'))

    # optimizer specials
    parser.add_argument(
        '--optimizer', help='Optimizer type for training models',
        default=None, type=str,
        choices=['rmsprop', None, 'adam', 'gd', 'sgd'])
    parser.add_argument(
        '--sgd_moment', help='moment decay for SGD',
        type=np.float32, default=0.9)
    parser.add_argument(
        '--rmsprop_epsilon', help='epsilon used by rmsprop',
        type=np.float32, default=None)
    parser.add_argument(
        '--lr_decay_method',
        help='type of learning rate decay. (None means user specified)',
        type=str, default=None,
        choices=[None, 'exponential', 'cosine', 'human'])
    parser.add_argument(
        '--init_lr_per_sample', help='Initial learning rate per sample',
        type=float, default=None)
    parser.add_argument(
        '--lr_decay', type=float, default=None,
        help='Amount of lr decay')
    parser.add_argument(
        '--lr_decay_every', type=float, default=None,
        help='Number of epochs per decay')

    # Placeholder: for internal use only do not accidently declare these.
    #parser.add_argument('--compute_hallu_stats')
    #parser.add_argument('--gradprocs')
    #parser.add_argument('--net_info')
    #parser.add_argument('--mlp_input_types') # and friend
    #parser.add_argument('--candidate_gate_eps') # used for training with sf/sg.
    # training param determined by dataset
    #parser.add_argument('--steps_per_epoch', type=int)
    #parser.add_argument('--max_epoch', type=int)
    parser.add_argument(
        '--max_train_steps', default=None, type=int,
        help="Do not specify. used for drop path internally")
    parser.add_argument(
        '--drop_path_keep_prob', default=1.0, type=float,
        help=('Keep probability for drop path. '
              '1.0 for mobile, 0.6 for cifar. 0.7 for imagenet'))
    parser.add_argument(
        '--dense_dropout_keep_prob', default=1.0, type=float,
        help='Keep probability for layer right before fc. 0.5 for imagenet')
    parser.add_argument(
        '--label_smoothing', default=0.0, type=float,
        help='label_smoothing param for cross_entropy. 0.1 for imagenet')

    ## Special options to force input types
    # and do mean/std process in graph in order to save memory
    # during cpu - gpu communication.
    # int32 is for embedding discrete inputs.
    parser.add_argument(
        '--input_type', type=str, default='float32',
        choices=['float32', 'uint8', 'int32'],
        help='Type for input, uint8 for certain dataset to speed up')
    parser.add_argument(
        '--output_type', type=str, default='int32',
        choices=['float32', 'int32'])
    parser.add_argument(
        '--do_mean_std_gpu_process',
        help='Whether use args.mean args.std to process in graph',
        default=False, action='store_true')

    ## alternative_training_target, distillation/compression
    parser.add_argument(
        '--alter_label', default=False, action='store_true',
        help="Type of alternative target to use")
    parser.add_argument(
        '--alter_loss_w', type=np.float32, default=0.5,
        help="percentage of alter loss weight")
    parser.add_argument(
        '--alter_label_activate_frac',
        help="Fraction of anytime predictions that uses alter_label",
        type=np.float32, default=0.75)
    parser.add_argument(
        '--high_temperature', type=np.float32, default=1.0,
        help='Temperature for training distill targets')
    parser.add_argument(
        '--hint_weight', type=np.float32, default=0,
        help="weight of hint loss")

    ## model_rnn specific
    # Since RNN models tend to have very different default params,
    # these params are specified by the training scripts
    parser.add_argument('--model_rnn_max_len', default=None, type=int)
    parser.add_argument('--model_rnn_vocab_size', default=None, type=int)
    parser.add_argument('--model_rnn_num_units', default=None, type=int)
    parser.add_argument('--model_rnn_num_lstms', default=None, type=int)
    parser.add_argument('--model_rnn_num_proj', default=None, type=int)
    parser.add_argument('--model_rnn_keep_prob', default=None, type=float)
    parser.add_argument('--model_rnn_init_range', default=None, type=float)
    parser.add_argument('--model_rnn_l2_reg', default=None, type=float)
    parser.add_argument('--model_rnn_slowness_reg', default=None, type=float)
    parser.add_argument(
        '--model_rnn_lock_embedding', default=False, action='store_true',
        help='Whether prediction embedding is the same as the input embedding')
    parser.add_argument(
        '--model_rnn_has_static_len', default=False, action='store_true')

    # Flags for debugging
    parser.add_argument('--debug_child_max_epoch', type=int, default=None)
    parser.add_argument('--debug_steps_per_epoch', type=int, default=None)

    parser.add_argument('--max_train_model_epoch', type=int, default=40)
    parser.add_argument('--init_model_epoch', type=int, default=None)
    parser.add_argument(
        '--do_remote_child_inf_runner', default=False, action='store_true')


def add_controller_arguments(parser):
    parser.add_argument(
        '--use_init_model', default=False, action='store_true')
    parser.add_argument(
        '--use_aux_head', default=False, action='store_true')
    parser.add_argument(
        '--use_local_reduction', default=False, action='store_true',
        help=('If downsample is needed on inputs, should the downsample of '
             'prev layer be global or local.'))
    parser.add_argument(
        '--use_hallu_feat_sel', default=False, action='store_true',
        help=('Whether to use feature selection to choose which '
              'operations to use on inputs'))

    parser.add_argument(
        '--num_init_use_all_gpu', default=1, type=int,
        help='number of initial jobs  that will use all gpus.')
    parser.add_argument(
        '--init_model_dir', type=str, default=None)
    parser.add_argument(
        '--init_blk_config',type=str, default=None,
        help='initial block configuration')
    parser.add_argument(
        '--init_n', default=None, type=str,
        help='comma separated list of int, which is num cells per reduction')
    parser.add_argument(
        '--max_growth', type=int, default=16,
        help='max iteration of growth; negative for forever (stop at mem error)')
    parser.add_argument(
        '--max_exploration', type=int, default=100000,
        help='Max number of children we are to explore, including hallu models')
    parser.add_argument(
        '--search_max_flops', type=float, default=None,
        help='max iteration of growth; negative for forever (stop at mem error)')
    parser.add_argument(
        '--moving_down_sampling', default=False, action='store_true',
        help='Whether move the down sampling layers as layers grow')
    parser.add_argument(
        '--prev_model_dir', default=None, type=str,
        help='directory from which the child is loading from')
    parser.add_argument(
        '--launch_remote_jobs', default=False, action='store_true')
    parser.add_argument(
        '--child_train_from_scratch', default=False, action='store_true')
    parser.add_argument(
        '--netmorph_method',
        default='hard', choices=['soft', 'hard', 'soft_gateback'])
    parser.add_argument(
        '--n_reduction_layers', type=int, default=2,
        help='number of reduction layers per resolution when constructing net')

    # Parameter for searching / hallucinations
    parser.add_argument(
        '--n_hallus_per_init', default=None, type=int,
        help='Minimum number of hallus to add to a model from q_parent')
    parser.add_argument(
        '--n_hallus_per_init_ratio', default=None, type=float,
        help='Minimum increase in compute divided by currect compute.')
    parser.add_argument(
        '--n_parent_reuses', default=-1, type=int,
        help=('Maximum number of times we can use a model in q_parent'
              ' (None is infinity). Negative or None for inifinite reuse'))
    parser.add_argument(
        '--n_hallus_per_select', default=None, type=int,
        help=('Number of hallus to select in a model from'
              ' q_hallu to form a model for q_child.'))
    parser.add_argument(
        '--n_hallus_per_select_ratio', default=None, type=float,
        help='Fraction of initialized hallus to select to form child.')
    parser.add_argument(
        '--n_rand_select_per_init', default=None, type=int,
        help='Number q_child models to randomly generate per q_hallu model.')
    parser.add_argument(
        '--n_greed_select_per_init', default=None, type=int,
        help=('Number q_child models to greedy select'
              '(based on statistics) per q_hallu model.'))
    parser.add_argument(
        '--hallu_final_merge_op', default=LayerTypes.MERGE_WITH_SUM, type=int,
        choices=[LayerTypes.MERGE_WITH_SUM, LayerTypes.MERGE_WITH_CAT,
            LayerTypes.MERGE_WITH_AVG, LayerTypes.MERGE_WITH_CAT_PROJ],
        help='int in the listed choices are sum, cat, avg, cat_proj')
    # Input choices
    parser.add_argument(
        '--hallu_input_choice', default=None, type=str,
        choices=['fix', 'rand', 'log', None],
        help='Method for choosing the x_in in hallucinations')
    # Operation feature selection parameter
    parser.add_argument(
        '--hallu_feat_sel_num', default=None, type=int,
        help='Number of operations to select if we use feature selection.')
    parser.add_argument(
        '--hallu_feat_sel_threshold', default=None, type=np.float32,
        help='Threshold on abs(omega) for selecting a feature(operation).')
    parser.add_argument(
        '--feat_sel_lambda', default=None, type=float,
        help='L1 lambda weight for feature selection'
    )

    # How to sort queues?
    parser.add_argument(
        '--q_parent_method', type=int, default=1,
        help='method for sorting q_parent. See petridish.nas_control.QueueSortMethods')
    parser.add_argument(
        '--q_hallu_method', type=int, default=1,
        help='method for sorting q_hallu. See petridish.nas_control.QueueSortMethods')
    parser.add_argument(
        '--q_child_method', type=int, default=1,
        help='method for sorting q_child. See petridish.nas_control.QueueSortMethods')

    # Queue sorting params
    parser.add_argument(
        '--q_parent_convex_hull_eps', type=float, default=None,
        help=('The hull can have an eps multiplative bandwidth'
            ' from the linear interp for convex hull relaxation'))
    parser.add_argument(
        '--do_convex_hull_increase', default=False,
        action='store_true', help='Whether allow convex hull to increase.')

    # parameterized controller option
    parser.add_argument(
        '--controller_type', type=int, default=0,
        help=('Type of controllers {{0: recognition, 1: MLP, '
              '2: RNN_SINGLE, 3: RNN_PER_STEP}}'))
    parser.add_argument(
        '--critic_type', type=int, choices=[0,1], default=0,
        help='critic type : {{0 : conv, 1 : lstm}}')
    parser.add_argument(
        '--critic_crawl_dirs', type=str, default=None,
        help='where the critic is going to look for data for training')
    parser.add_argument(
        '--critic_train_epoch', type=int, default=40,
        help='num epochs to train the critic')
    parser.add_argument(
        '--critic_init_lr', type=float, default=1e-3)

    parser.add_argument('--controller_seq_length', type=int, default=None)
    parser.add_argument('--controller_max_depth', type=int, default=128)
    parser.add_argument('--controller_batch_size', type=int, default=1)
    parser.add_argument('--controller_dropout_kp', type=float, default=1)
    parser.add_argument('--controller_train_every', type=int, default=10000)
    parser.add_argument('--controller_save_every', type=int, default=1)

    parser.add_argument(
        '--search_cell_based', default=False, action='store_true',
        help='Whether searches cells (True) or overall network (False)')
    parser.add_argument(
        '--search_cat_based', default=False, action='store_true',
        help=('Whether the candidate layers are sum (False/default)'
            ' or cat (True) with target layer'))

    parser.add_argument('--init_cell_type', type=str, default=None)

    parser.add_argument('--lstm_size', type=int, default=32)
    parser.add_argument('--num_lstms', type=int, default=2)

    parser.add_argument('--n_hallu_procs_per_gpu', type=float, default=1)
    parser.add_argument('--n_model_procs_per_gpu', type=float, default=1)
    parser.add_argument('--n_critic_procs_per_gpu', type=float, default=1)

    parser.add_argument(
        '--grow_petridish_version', type=str, default='mp',
        choices=['mp', 'soft_vs_hard', 'inc_vs_scratch'])

    # Multi-process speciall options: "Remote" options:
    # Determinte roles "Server", "Remote"
    parser.add_argument(
        '--job_type', help='type of operation this job is',
        default='main', choices=['main', 'remote_child', 'remote_critic'])
    # Remote critic specific
    parser.add_argument(
        '--queue_name', type=str, default=None,
        help="Used by remote critic worker to determine which queue to train")
    parser.add_argument(
        '--store_critic_data', default=False, action='store_true')
    # Server specific for killing crawlers if given.
    parser.add_argument(
        '--companion_pids', type=str, default=None,
        help='A comma-separated list PID to kill at exit by server (job_type==main).')
    parser.add_argument(
        '--launch_local_crawler', default=False, action='store_true')


def scale_int_val_with_gpu(val_per, nr_gpu):
    return max(int(val_per * nr_gpu + 0.5), 1)

def model_options_processing(options):
    """
    Populate some complicated default arguments, and parse comma-separated int lists to int lists.
    """
    if options.net_info_str is None:
        options.net_info = None
        return options
    if isinstance(options.net_info_str, str):
        try:
            options.net_info = net_info_from_str(options.net_info_str)
        except:
            logger.info("Failed info str is:\n{}".format(options.net_info_str))
            raise
    return options

def is_debug(options):
    return (options.debug_child_max_epoch is not None or
        options.debug_steps_per_epoch is not None)

def options_to_str(options, ignore=[]):
    ss = ""
    for key in ignore:
        if key == 'data_dir':
            ss += "--data_dir=$DATA_DIR \\\n"
        elif key == 'log_dir':
            ss += "--log_dir=$LOG_DIR \\\n"
        elif key == 'model_dir':
            ss += "--model_dir=$MODEL_DIR \\\n"
        elif key == 'load':
            ss += "--load=$MODEL_DIR/checkpoint \\\n"

    arg_dict = vars(options)
    keys = sorted(arg_dict.keys())
    for key in keys:
        if key in ignore:
            continue
        val = arg_dict[key]
        if isinstance(val, bool):
            if val:
                ss += '--{key} \\\n'.format(key=key)
            continue

        if val is None:
            continue

        if isinstance(val, str) and val == "":
           ss += '--{key}="" \\\n'.format(key=key)
           continue

        if isinstance(val, list):
            if len(val) == 0:
                val_str = "\"\""
            elif isinstance(val[0], int) or isinstance(val[0], float):
                val_str = ','.join(map(str, val))
            elif isinstance(val[0], str):
                val_str = ','.join(val)
            elif isinstance(val[0], LayerInfo):
                val_str = "\'{}\'".format(val.to_str())
                key = key + '_str'
        else:
            val_str = str(val)

        if isinstance(val, dict):
            if len(val) == 0:
                val_str = "\"\""
            elif isinstance(val, CellNetworkInfo):
                val_str = "\'{}\'".format(val.to_str())
                key = key + '_str'
        ss += '--{key}={val} \\\n'.format(key=key, val=val_str)
    return ss

def generate_code_record_options(prefix='', vnames=[]):
    """
    e.g.,
    generate_code_record_options('model_rnn_',
        ['max_len', 'vocab_size', 'num_units', 'num_lstms', 'num_proj', 'keep_prob'])
    """
    code = []
    for vname in vnames:
        code.append('self.{vname} = options.{prefix}{vname}'.format(vname=vname, prefix=prefix))
    return '\n\t\t'.join(code)