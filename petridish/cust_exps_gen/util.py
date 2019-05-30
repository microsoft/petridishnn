import argparse
import itertools
import copy
import os
import numpy as np

import anytime_models.examples.ann_app_utils as ann_app_utils
from petridish.philly.generator import SCRIPT_TEMPLATE
from petridish.app.options import (
    add_app_arguments, add_model_arguments, add_controller_arguments,
    options_to_str)
from petridish.info import (
    net_info_from_str, add_aux_weight, net_info_cifar_to_ilsvrc,
    replace_wsum_with_catproj, increase_net_info_size)
from petridish.nas_control.controller import QueueSortMethods, ControllerTypes
from petridish.nas_control.critic import CriticTypes
from petridish.data.openml import cbb_openml_indices, cbb_openml_indices_failed


pre_entry_cmds = \
"""
env
"""

def form_default_args():
    parser = argparse.ArgumentParser()
    add_app_arguments(parser)
    add_model_arguments(parser)
    add_controller_arguments(parser)
    args, _unknown = parser.parse_known_args()
    #print("Has unknown args : {}".format(unknown))
    return args

def form_cust_exp_bash(
        batch_desc, gen_options, eidx, bidx,
        entry='petridish/app/petridish_main.py', debug=False):
    options = form_default_args()
    log_str = [ 'Experiment {} : {}'.format(bidx, batch_desc) ]
    eidx +=1
    for options, desc in gen_options(options):
        eidx +=1
        log_str.append('  {} : {}'.format(eidx, desc))
        ignore_list = ['data_dir', 'log_dir', 'model_dir']
        ostr = options_to_str(options, ignore=ignore_list)
        script_str = SCRIPT_TEMPLATE.format(entry=entry, options=ostr,
            pre_entry_cmds=pre_entry_cmds)
        if not debug:
            with open(os.path.join('cust_exps', 'petridish_{}.sh'.format(eidx)), 'wt') as fout:
                fout.write(script_str)
    print('\n'.join(log_str))
    return eidx

def cifar_default_search_options(options):
    # aux prediction related. Legacy. keep constant and ignore
    options.adaloss_final_extra=0.0
    options.adaloss_gamma=0.07
    options.adaloss_momentum=0.9
    options.adaloss_update_per=100

    # distilled targets. Legacy. keep constant and ignore
    options.alter_label_activate_frac=0.75
    options.alter_loss_w=0.5
    # basic vs bottleneck. Legacy. keep constant and ignore
    options.b_type='basic'
    # batch norm param.
    options.batch_norm_decay=0.9
    options.batch_norm_epsilon=1e-05
    # batch size PER GPU
    options.batch_size_per_gpu=32

    # Legacy critic/constroller training. unused. ignore
    options.controller_batch_size=1
    options.controller_dropout_kp=0.5
    options.controller_max_depth=128
    options.controller_save_every=1
    options.controller_train_every=10000
    options.controller_type=0
    options.critic_init_lr=0.001
    options.critic_train_epoch=40
    options.critic_type=1

    # chanels_first vs. channels_last
    options.data_format='channels_first'
    # final feature drop out
    options.dense_dropout_keep_prob=1.0
    # search always uses val set. keep constant
    options.do_validation = True
    # Drop path
    options.drop_path_keep_prob=0.6
    # dataset name
    options.ds_name='cifar10'
    # multi-process version.
    options.grow_petridish_version='mp'
    # number of operations to be jointed together to form a weak learner
    options.hallu_feat_sel_num=3
    # weight threshold to keep an operation
    options.hallu_feat_sel_threshold=0.0
    # how weak learner and parent model are joined together.
    # 11 means sum. See petridish/info/layer_info.py
    options.hallu_final_merge_op=11
    # Prediction temperature. Used for distillation. 1.0 means no regulation.
    options.high_temperature=1.0
    # Legacy critic/constroller training. unused. ignore
    options.hint_weight=0
    # initial channel size.
    options.init_channel=16
    # initial learning rate PER SAMPLE
    options.init_lr_per_sample= 0.00078125
    # Number of epochs to train the seed model
    options.init_model_epoch=200
    # Number of normal cells per resolution
    options.init_n=3
    # height and width of inputs
    options.input_size=32
    # dtype of input
    options.input_type='float32'
    # this indicate to do search. Keep constant
    options.job_type='main'
    # label smoothing regulation. 0.0 means none.
    options.label_smoothing=0.0
    # whether to launch the companion that launches children training
    # Always True for search
    options.launch_local_crawler = True
    # Aux prediction weight balancing method. Keep constantant.
    options.ls_method=100
    # Legacy critic/constroller training. unused. ignore
    options.lstm_size=32
    # Max number of models to search
    options.max_exploration=2000
    # Max deviation from the original seed model.
    # This means the max search depth will be 16.
    options.max_growth=8
    # Number of eoochs per child/weak learner training.
    options.max_train_model_epoch=80
    # Legacy. Unused. Ignore.
    options.n_critic_procs_per_gpu= 1.0 / options.nr_gpu
    # Legacy. Unused. Ignore.
    options.n_greed_select_per_init=0

    # number of weak learner training per GPU when we do multi-process
    options.n_hallu_procs_per_gpu=0.5
    # Always 1 for cell search
    options.n_hallus_per_init=1
    options.n_hallus_per_select=1

    # number of children training per GPU when we do multi-process
    options.n_model_procs_per_gpu=1
    # Can reuse infinitely many times
    options.n_parent_reuses=-1
    # number of weak learners to select.
    # 1 because there will be only one choice
    options.n_rand_select_per_init=1
    # Number of reduction layers
    options.n_reduction_layers=2
    # Whether to block influence from weak learner to parent during
    # initial training of weak learner. hard means block. soft means no block.
    options.netmorph_method='hard'

    options.nr_gpu=4
    options.num_classes=10

    # number of initital models that will use all GPU
    options.num_init_use_all_gpu=1
    # Legacy. unused. Ignore
    options.num_lstms=2
    # prediction type
    options.output_type='int32'
    # prediction preprocess.
    options.prediction_feature='1x1'
    # first in last out on children and weak learner training queue
    # last-out because depth first search is generally better here.
    options.q_child_method=5
    options.q_hallu_method=5
    # relaxation on convex hull
    options.q_parent_convex_hull_eps=0.025
    # How to select parent model. See petridish/nas_control/controller.py
    options.q_parent_method=7
    # regularization on weights of model
    options.regularize_coef='const'
    options.regularize_const=0.0001
    # starting operations. See petridish/model/layer.py function initial_convolution
    options.s_type='basic'
    # Cell vs. macro search
    options.search_cell_based = True
    # maximum FLOPS allowed to be a parent model.
    options.search_max_flops=166666666.667
    # SGD momentum
    options.sgd_moment=0.9
    # Multiplier on the channel size for the first conv.
    options.stem_channel_rate=3.0
    # Whether use feature selection. Always True
    options.use_hallu_feat_sel = True
    # Model weight initialization method
    options.w_init='var_scale'
    return options

def cifar_default_train_options(options):
    options.job_type = 'remote_child'
    options.batch_size_per_gpu = 32
    options.init_lr_per_sample = 0.1 / 128
    options.label_smoothing = 0.0
    options.ls_method = 0
    options.stem_channel_rate = 3.0
    options.init_channel = 32
    options.drop_path_keep_prob = 0.6
    options.max_train_model_epoch = 600
    # Use train/test split, instead of spliting part of train to train/val.
    options.do_validation = False
    options.regularize_coef = 'const'
    options.regularize_const = 5e-4
    # Whether to evaluate every epoch
    options.do_remote_child_inf_runner = True
    options.nr_gpu = 1
    return options

def imagenet_mobile_default_train_options(options):
    # batch size (per replica): 32
    # learning rate: 0.04 * 50
    # learning rate scaling factor: 0.97
    # num epochs per decay: 2.4
    # sync sgd with 50 replicas
    # auxiliary head weighting: 0.4
    # label smoothing: 0.1
    # clip global norm of all gradients by 10
    options.job_type = 'remote_child'
    options.batch_size_per_gpu = 32
    options.init_lr_per_sample = 0.04 / 32
    options.label_smoothing = 0.1
    options.ls_method = 0
    options.dense_dropout_keep_prob = 1.0 # 0.55
    # channels in stem reductions.

    # these two are changed because of extra reduction layers.
    n_extra_reductions = 2
    options.stem_channel_rate = 1.0 * 32. / options.init_channel
    options.do_validation = False
    #options.regularize_coef = 'const'
    #options.regularize_const = 4e-5
    #options.batch_norm_decay = 0.9997
    #options.batch_norm_epsilon = 1e-3
    options.max_train_model_epoch = 250
    options.s_type = 'conv3'
    options.b_type = 'bottleneck'
    options.do_remote_child_inf_runner = True
    options.training_type = 'darts_imagenet'
    options.drop_path_keep_prob = 0.7
    options.init_channel = 11
    options.nr_gpu = 4
    return options

def model_desc_fn_to_option(
        fullname, ds_name, options,
        use_latest_input=False,
        aux_weight=0.4,
        depth_multiplier=[1]):
    assert os.path.exists(fullname), fullname
    with open(fullname, 'rt') as fin:
        lines = fin.readlines()
        assert lines, 'file is empty'
        line = lines[0].strip()
        assert line, 'the net info line (first line) is empty'

    options.ds_name = ds_name
    if ds_name == 'cifar10' or ds_name == 'cifar100':
        is_ilsvrc = False
        options = cifar_default_train_options(options)
    elif ds_name == 'imagenet' or ds_name == 'ilsvrc':
        is_ilsvrc = True
        options = imagenet_mobile_default_train_options(options)

    net_info = net_info_from_str(line)
    if isinstance(depth_multiplier, int):
        depth_multiplier = [depth_multiplier]
    if any([_x > 1 for _x in depth_multiplier]):
        net_info = increase_net_info_size(
            net_info, depth_multiplier)
    if is_ilsvrc:
        net_info = net_info_cifar_to_ilsvrc(
            net_info, options.s_type, use_latest_input)
    if aux_weight > 0:
        options.net_info = add_aux_weight(net_info, 0.4)
    return options

def generate_train_options(
        net_info_fn,
        ds_name,
        depth_multiplier=2):
    """
    Given a file containing a single line of net_info,
    produces a default training options

    Args:
    net_info_fn (str): path to a file that contains a net_info_str
    ds_name (str): dataset either cifar10, cifar100, or ilsvrc.
    depth_multiplier (int or list of int):
        The number of times a cell is multiplied.
        If it is an int, it is converted to [depth_multiplier] first.
        If it is a list, it means k-th normal cell will appear
            depth_multiplier[k % len(depth_multiplier)] times
            instead of once.

    Returns:
    An options object of the training, and a description str
    of the option.
    """
    options = form_default_args()

    # this will change the 9 normal cell into 18 normal cell.
    options = model_desc_fn_to_option(
        net_info_fn,
        ds_name,
        options,
        use_latest_input=True,
        depth_multiplier=depth_multiplier)
    # ablation studies show that we should replace the weigthed sum
    # for feature(operation) selection with concat-projection
    options.net_info = replace_wsum_with_catproj(options.net_info)
    desc = '{} training with {}'.format(ds_name, net_info_fn)
    return options, desc

def options_to_script_str(options):
    ostr = options_to_str(options, ignore=['data_dir', 'log_dir', 'model_dir'])
    entry = 'petridish/app/petridish_main.py'
    script_str = SCRIPT_TEMPLATE.format(
        entry=entry, options=ostr,
        pre_entry_cmds=pre_entry_cmds)
    return script_str