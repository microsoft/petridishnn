# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import tensorflow as tf
import os
import argparse
import copy
import subprocess

from tensorpack.train import (TrainConfig, DEFAULT_CALLBACKS,
    SyncMultiGPUTrainerParameterServer, launch_train_with_config)
from tensorpack.tfutils.sessinit import (
    SaverRestore, get_model_loader, ChainInit)
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.tfutils.gradproc import GlobalNormClip
from tensorpack.callbacks import (ModelSaver, InferenceRunner, ScalarStats,
    ScheduledHyperParamSetter, HumanHyperParamSetter,
    JSONWriter, ScalarPrinter, TFEventWriter, CallbackFactory,
    ProgressBar, RunOp, HyperParamSetterWithFunc, PeriodicTrigger)
from tensorpack.input_source import TensorInput
from tensorpack.dataflow.common import FixedSizeData
from tensorpack.utils.stats import RatioCounter, StatCounter
from tensorpack.predict import PredictConfig, OfflinePredictor
from tensorpack.utils import logger, utils, fs, stats

import anytime_models.examples.ann_app_utils as ann_app_utils
import anytime_models.examples.get_augmented_data as get_augmented_data

from petridish.app.options import scale_int_val_with_gpu
from petridish.info import LayerInfoList, LayerTypes, LayerInfo
from petridish.model import (
    generate_classification_callbacks, generate_regression_callbacks,
    RecognitionModel, MLPModel, DYNAMIC_WEIGHTS_NAME,
    get_hallu_stats_output_names, get_feature_selection_weight_names,
    get_net_info_hallu_stats_output_names
)
from petridish.data import (tiny_imagenet, downsampled_imagenet, openml,
    speech_commands, ptb, cifar, imagenet, inat)
from petridish.data.misc import (preprocess_data_flow, get_csv_data,
    LoadedDataFlow, AgeOnlyCallBack)
from petridish.utils.sessinit import (
    SaverRestoreSizeRelaxed, read_parameter_val, AssignGlobalStep)
from petridish.utils.callbacks import (
    PerStepHookWithControlDependencies,
    PerStepInferencer)

def get_training_params(model_cls, args, is_training=True):
    """
    Data set specific params. Modify args for the specific data-set.
    """
    model = None
    ds_train, ds_val, insrc_train, insrc_val = None, None, None, None
    args.steps_per_epoch = None
    lr_schedule = None
    has_cbs_init = False
    train_cbs = []
    val_cbs = []
    output_names = None
    output_funcs = None

    args.batch_size = scale_int_val_with_gpu(args.batch_size_per_gpu, args.nr_gpu)
    args.init_lr = args.init_lr_per_sample * args.batch_size
    if args.ds_name == 'cifar10' or args.ds_name == 'cifar100':
        if args.ds_name == 'cifar10':
            args.num_classes = 10
        else:
            args.num_classes = 100
        args.regularize_coef = 'decay'
        args.input_size = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = cifar.get_cifar_augmented_data

        if is_training:
            ds_train = get_data(
                'train', args, do_multiprocess=True,
                do_validation=args.do_validation, shuffle=True)
            if args.training_type == 'darts_cifar':
                args.init_lr = 0.025
                args.regularize_coef = 'const'
                args.regularize_const = 3e-4

            lr = float(args.init_lr)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = args.init_model_epoch
            else:
                max_epoch = args.max_train_model_epoch
            max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.lr_decay_method = 'cosine'
            args.gradprocs = [GlobalNormClip(5)]
            args.max_epoch = max_epoch
            args.steps_per_epoch = ds_train.size()

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = get_data(
                'test', args, do_multiprocess=False,
                do_validation=args.do_validation, shuffle=False)

    elif args.ds_name == 'ilsvrc' or args.ds_name == 'imagenet':
        args.num_classes = 1000
        args.input_size = 224

        args.do_mean_std_gpu_process = True
        args.input_type = 'uint8'
        args.mean = imagenet.ilsvrc_mean
        args.std = imagenet.ilsvrc_std
        #args.s_type = 'imagenet' # make sure to check this...

        get_data = imagenet.get_ilsvrc_augmented_data
        if is_training:
            ds_train = get_data(
                'train', args, do_multiprocess=True, is_train=True, shuffle=True)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 100
            else:
                max_epoch = args.max_train_model_epoch
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            args = imagenet.training_params_update(args)
            args.gradprocs = [GlobalNormClip(5)]

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = get_data(
                'val', args, do_multiprocess=True, is_train=False, shuffle=True)

    elif args.ds_name == 'tiny_imagenet':
        # fix data-set specific params
        args.num_classes = 200
        args.input_size = 64

        # transfer uint8 data and cast to float in gpu
        args.do_mean_std_gpu_process = True
        args.input_type = 'uint8'
        args.mean = get_augmented_data.ilsvrc_mean
        args.std = get_augmented_data.ilsvrc_std
        args.s_type = 'conv7'
        args.b_type = 'bottleneck'

        # training params
        args.regularize_coef = 'const'
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = tiny_imagenet.get_tiny_imagenet_augmented_data

        if is_training:
            ds_train = get_data('train', args, do_multiprocess=True, shuffle=True, is_train=True)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 100
            else:
                max_epoch = args.max_train_model_epoch
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            args = imagenet.training_params_update(args)
            args.gradprocs = [GlobalNormClip(10)]

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = get_data('val',  args, do_multiprocess=True, is_train=False)

    elif downsampled_imagenet.is_ds_name_downsampled_imagenet(args.ds_name):
        args.num_classes = 1000
        args.input_size = downsampled_imagenet.ds_name_to_input_size(args.ds_name)
        args.regularize_coef = 'decay'
        args.b_type = 'bottleneck'

        get_data = downsampled_imagenet.get_downsampled_imagenet_augmented_data
        if is_training:
            ds_train = get_data('train', args, do_multiprocess=True, shuffle=True,
                do_validation=args.do_validation)

            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 100
            else:
                max_epoch = args.max_train_model_epoch
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            args = imagenet.training_params_update(args)
            args.gradprocs = [GlobalNormClip(10)]

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = get_data('val',  args, do_multiprocess=True,
                do_validation=args.do_validation)

    elif args.ds_name == 'speech_commands':
        args.regularize_coef = 'const'
        args.num_classes = len(speech_commands.DEFAULT_TRAIN_WORDS) + 2
        get_data = speech_commands.get_augmented_speech_commands_data

        if is_training:
            ds_train = get_data('train', args, do_multiprocess=True, shuffle=True)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 90
            else:
                max_epoch = args.max_train_model_epoch
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            args = imagenet.training_params_update(args)
            args.gradprocs = [GlobalNormClip(10)]

        if args.do_remote_child_inf_runner or not is_training:
            val_split = 'val' if args.do_validation else 'test'
            ds_val = get_data(val_split, args, do_multiprocess=False, shuffle=False)

    elif args.ds_name == 'svhn':
        args.num_classes = 10
        args.regularize_coef = 'decay'
        args.input_size = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_augmented_data.get_svhn_augmented_data

        ## Training model
        if is_training:
            ds_train = get_data('train', args, do_multiprocess=True, shuffle=True)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 60
            else:
                max_epoch = 12
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            args.lr_decay_method = 'cosine'
            args.gradprocs = [GlobalNormClip(5)]

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = get_data('test', args, do_multiprocess=False, shuffle=False)

    elif args.ds_name.startswith('openml_'):
        int_start = args.ds_name.find('_') + 1
        dataset_idx = int(args.ds_name[int_start:])
        # Some arg protetection in case these are used in the future
        #assert not hasattr(args, 'mlp_input_types') and not hasattr(args, 'mlp_input_dims')
        (l_ds, args.mlp_input_types,
            args.mlp_input_dims, n_data, args.num_classes,
            args.mlp_feat_means, args.mlp_feat_stds
        ) = openml.get_openml_dataflow(
            dataset_idx, args.data_dir, splits=['train','val'],
            do_validation=args.do_validation
        )
        ds_train = preprocess_data_flow(l_ds['train'], args, True)
        ds_val = preprocess_data_flow(l_ds['val'], args, False)
        logger.info("Dataset {} has {} samples and {} dims".format(\
            args.ds_name, n_data, len(args.mlp_input_types)))

        if is_training:
            lr = float(args.init_lr)
            lr = float(args.init_lr)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = args.init_model_epoch
            else:
                max_epoch = args.max_train_model_epoch

            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            max_epoch = args.max_epoch
            lr_schedule = [
                (1, lr),
                (max_epoch // 2, lr * 1e-1),
                (max_epoch * 3 // 4, lr * 1e-2),
                (max_epoch * 7 // 8, lr * 1e-3)
            ]

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = preprocess_data_flow(l_ds['val'], args, True)

    elif args.ds_name == 'inat' or args.ds_name == 'inat100' or args.ds_name == 'inat1000' or args.ds_name == 'inat2017_1000':
        inat_lmdb_dir = None
        inat_year = '2018'

        if args.ds_name == 'inat':
            args.num_classes = 8142
            n_allow = None
        elif args.ds_name == 'inat100':
            args.num_classes = 100
            n_allow = 100
        elif args.ds_name == 'inat1000':
            args.num_classes = 1000
            n_allow = 1000
        elif args.ds_name == 'inat2017_1000':
            args.num_classes = 1000
            n_allow = 1000
            inat_year = '2017'
            inat_lmdb_dir = 'inat2017_data/lmdb'
        args.input_size = 224

        args.do_mean_std_gpu_process = True
        args.input_type = 'uint8'
        args.mean = inat.image_mean
        args.std = inat.image_std

        get_data = inat.get_inat_augmented_data
        if is_training:
            ds_train = get_data(
                'train',
                args,
                lmdb_dir=inat_lmdb_dir, year=inat_year,
                do_multiprocess=True,
                do_validation=args.do_validation,
                is_train=True,
                shuffle=True,
                n_allow=n_allow)
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 100
            else:
                max_epoch = args.max_train_model_epoch
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.steps_per_epoch = ds_train.size()
            args = imagenet.training_params_update(args)
            args.gradprocs = [GlobalNormClip(5)]

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = get_data(
                'val',
                args,
                lmdb_dir=inat_lmdb_dir, year=inat_year,
                do_multiprocess=True,
                do_validation=args.do_validation,
                is_train=False,
                shuffle=True,
                n_allow=n_allow)


    elif args.ds_name == 'ptb':
        ptb_data_dir = os.path.join(args.data_dir, 'ptb_data')
        args.input_type = 'int32'
        args.search_cell_based = False
        # force single gpu for now.
        args.nr_gpu = 1
        args, local_args = ptb.training_params_update(args)

        # evaluation/testing batch size change.
        if not is_training:
            if args.do_validation:
                args.batch_size_per_gpu = 64
            else:
                args.batch_size_per_gpu = 64
        # update globale batch size.
        args.batch_size = args.batch_size_per_gpu * args.nr_gpu
        args.init_lr = args.init_lr_per_sample * args.batch_size

        if is_training:
            var_size = not args.model_rnn_has_static_len
            ds_train = ptb.PennTreeBankDataFlow(
               'train',
               ptb_data_dir,
               args.batch_size,
               args.model_rnn_max_len,
               var_size=var_size)
            args.steps_per_epoch = ds_train.size()
            args.model_rnn_vocab_size = ds_train.vocab_size
            if args.child_train_from_scratch and args.job_type == 'remote_child':
                max_epoch = 100
            else:
                max_epoch = args.max_train_model_epoch
            args.max_epoch = (max_epoch + args.nr_gpu - 1) // args.nr_gpu
            args.gradprocs = [GlobalNormClip(local_args.grad_clip)]

            model = model_cls(args)
            #
            # compute some callbacks for training
            # We need to consturct the model now, as some op requires
            # the graph to be up. (shifting states and reset states)
            ptb.ptb_training_cbs(model, args, ptb_data_dir, train_cbs)
            has_cbs_init = True

        if args.do_remote_child_inf_runner or not is_training:
            ds_val = ptb.PennTreeBankDataFlow(
               'valid' if args.do_validation else 'test',
               ptb_data_dir,
               args.batch_size,
               args.model_rnn_max_len,
               var_size=False)
            args.model_rnn_vocab_size = ds_val.vocab_size

            model = model_cls(args)
            # testing set up.
            # log loss of each sample
            output_names = [
                #model.inference_update_tensor(name_only=True) + ':0',
                'avg_batch_cost:0',
                'seq_len:0',
                'per_seq_sum_logloss:0',
            ]
            # the averaging is done automatically over the batches
            # We need to average over the time.
            # We exponentiate per prediction logloss for the perplexity score.
            output_funcs = [
                #None,
                lambda x: x * args.batch_size,
                lambda x: x * args.batch_size,
                lambda x: np.exp(x / args.model_rnn_max_len),
            ]

    else:
        raise Exception("Unknown dataset {}".format(args.ds_name))

    # computing epochs / steps / reading init learning rate.
    # Last section that may affect the args
    args.max_train_steps = None
    if is_training:
        args.candidate_gate_eps = 1.0 / args.steps_per_epoch / args.batch_size
        starting_epoch = 1
        if args.model_dir is not None:
            ckpt = tf.train.latest_checkpoint(args.model_dir)
            if ckpt:
                starting_epoch = ann_app_utils.grep_starting_epoch(
                    ckpt, args.steps_per_epoch)
        if lr_schedule:
            args.init_lr = ann_app_utils.grep_init_lr(starting_epoch, lr_schedule)
        if args.debug_child_max_epoch:
            args.max_epoch = args.debug_child_max_epoch
        if args.debug_steps_per_epoch:
            args.steps_per_epoch = args.debug_steps_per_epoch
            starting_epoch = 1
        logger.info("Start at epoch {} with learning rate {}".format(
            starting_epoch, args.init_lr))
        args.max_train_steps = args.steps_per_epoch * args.max_epoch
        if model is not None:
            model.options.max_train_steps = args.max_train_steps

    if model is None:
        # if the dataset specific does not init the model, we init it here
        model = model_cls(args)
    # From now on args should be const.

    if is_training:
        if not has_cbs_init:
            if ds_val and args.debug_steps_per_epoch:
                ds_val = FixedSizeData(ds_val, args.debug_steps_per_epoch)
            train_cbs.extend(
                _inference_runner_train_cbs(args, ds_val, insrc_val, val_cbs))
        return (
            model,
            args,
            starting_epoch,
            lr_schedule,
            ds_train,
            insrc_train,
            train_cbs
        )

    else:
        if output_names is None:
            output_names = _inference_output_names(args)
            output_funcs = [None] * len(output_names)
        return (
            model,
            args,
            ds_val,
            insrc_val,
            output_names,
            output_funcs
        )


def _inference_runner_train_cbs(args, ds_val, insrc_val, val_cbs):
    train_cbs = []
    if args.do_remote_child_inf_runner:
        if args.num_classes > 1:
            val_cbs.extend(
                generate_classification_callbacks(args.net_info.master))
        else:
            val_cbs.extend(
                generate_regression_callbacks(args.net_info.master))
        inf_runner = InferenceRunner(ds_val or insrc_val, [ScalarStats('cost')] + val_cbs)
        train_cbs.append(inf_runner)
    return train_cbs


def _inference_output_names(args):
    if args.num_classes > 1:
        output_names = generate_classification_callbacks(
            args.net_info.master, name_only=True)
    else:
        output_names = generate_regression_callbacks(
            args.net_info.master, name_only=True)
    return output_names

def train_child(model_cls, args, log_dir, child_dir, prev_dir):
    """
    """
    if not os.path.exists(child_dir):
        os.mkdir(child_dir)

    if os.path.basename(child_dir) == "0" and args.use_init_model:
        init_model_dir = os.path.join(args.data_dir, 'init_model', args.ds_name)
        if os.path.exists(init_model_dir):
            # This implies that there exists init_model_dir, and we are in first model
            # so we do not need to train. Copy the model and mark finished
            logger.info("Skip first model as this model is fully trained.")
            cmd = "mkdir -p {cdir} ; cp {pdir}/* {cdir}/ ".format(\
                cdir=child_dir, pdir=args.init_model_dir)
            _ = subprocess.check_output(cmd, shell=True)
            return

    # get training params for train-config
    (
        model, args, starting_epoch, lr_schedule,
        ds_train, insrc_train, train_cbs
    ) = get_training_params(model_cls, args)

    ## Model callbacks
    # loss weight update
    ls_cbs_func = getattr(model, 'compute_loss_select_callbacks', None)
    if callable(ls_cbs_func):
        train_cbs.extend(ls_cbs_func())
    # extra callback for general logging/ update.
    extra_callbacks = DEFAULT_CALLBACKS()
    if not args.do_remote_child_inf_runner:
        extra_callbacks = \
            [ecb for ecb in extra_callbacks if not isinstance(ecb, ProgressBar)]
    logger.info("Extra callbacks are {}".format(
        [ecb.__class__ for ecb in extra_callbacks]))

    # Logging for analysis
    model_str = model.net_info.to_str()
    logger.info('LayerInfoListString is :\n {}'.format(model_str))
    train_callbacks = [
        ModelSaver(
            checkpoint_dir=child_dir,
            max_to_keep=1,
            keep_checkpoint_every_n_hours=100
        ),
    ] + train_cbs
    if lr_schedule:
        train_callbacks.append(
            ScheduledHyperParamSetter('learning_rate', lr_schedule))
    logger.info('The updated params for training is \n{}'.format(args))
    config = TrainConfig(
        data=insrc_train,
        dataflow=ds_train,
        callbacks=train_callbacks,
        extra_callbacks=extra_callbacks,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()], #, TFEventWriter()],
        steps_per_epoch=args.steps_per_epoch,
        max_epoch=args.max_epoch,
        starting_epoch=starting_epoch
    )
    for dn in [child_dir, prev_dir]:
        if dn is None:
            continue
        ckpt = tf.train.latest_checkpoint(dn)
        if ckpt:
            if args.search_cat_based:
                restore_cls = SaverRestoreSizeRelaxed
            else:
                restore_cls = SaverRestore
            _ignore = [DYNAMIC_WEIGHTS_NAME]
            _sess_init_load = restore_cls(ckpt, ignore=_ignore)
            if dn == child_dir:
                # loading from self keep global step
                config.session_init = _sess_init_load
            else:
                # loading from others. Set global_step to 0
                config.session_init = ChainInit(
                    [
                        _sess_init_load,
                        AssignGlobalStep(0),
                    ])
            break
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(args.nr_gpu))
    return model

def eval_child(model_cls, args, log_dir, model_dir, collect_hallu_stats=True):
    """
    Args:
        model_cls (PetridishModel) :
        args :
        log_dir (str): where to log
        model_dir (str) : where to load from
        collect_hallu_stats (bool) : whether to collect hallu stats if there are any.
    Return:
        eval_vals (list) : a list of evaluation related value.
        The first is the vaildation error on the specified validation set;
        it is followed by hallucination stats.
    """
    ckpt = tf.train.latest_checkpoint(model_dir)
    if not ckpt:
        logger.info("No model exists. Do not sort")
        return []
    args.compute_hallu_stats = True
    (
        model, args, ds_val, insrc_val, output_names, output_funcs
    ) = get_training_params(model_cls, args, is_training=False)
    n_outputs = len(output_names)
    logger.info("{} num vals present. Will use the final perf {} as eval score".format(\
        n_outputs, output_names[-1]))
    stats_handlers = [StatCounter() for _ in range(n_outputs)]

    # additional handlers for hallucinations
    if collect_hallu_stats:
        hallu_stats_names = get_net_info_hallu_stats_output_names(model.net_info)
        stats_handlers.extend([StatCounter() for _ in hallu_stats_names])
        output_names.extend(hallu_stats_names)
    # Note at this point stats_handlers[n_outputs-1:] contains all
    # the value needed for evaluation.

    # batch size counter
    sample_counter = StatCounter()
    # ignore loading certain variables during inference
    ignore_names = getattr(model, 'load_ignore_var_names', [])
    pred_config = PredictConfig(
        model=model,
        input_names=model._input_names,
        output_names=output_names,
        session_init=SaverRestore(ckpt, ignore=ignore_names)
    )
    predictor = OfflinePredictor(pred_config)

    # two types of input, dataflow or input_source
    if ds_val:
        gen = ds_val.get_data()
        ds_val.reset_state()
        input_sess = None
    else:
        if not insrc_val.setup_done():
            insrc_val.setup(model.get_inputs_desc())
        sess_config = get_default_sess_config()
        sess_config.device_count['GPU'] = 0
        input_tensors = insrc_val.get_input_tensors()
        sess_creater = tf.train.ChiefSessionCreator(config=sess_config)
        input_sess = tf.train.MonitoredSession(sess_creater)
        def _gen_func():
            insrc_val.reset_state()
            for _ in range(insrc_val.size()):
                yield input_sess.run(input_tensors)
        gen = _gen_func()

    for dp_idx, dp in enumerate(gen):
        output = predictor(*dp)
        batch_size = output[n_outputs - 1].shape[0]
        sample_counter.feed(batch_size)
        for o, handler in zip(output, stats_handlers):
            handler.feed(np.sum(o))
        if (args.debug_steps_per_epoch and
                dp_idx + 1 >= args.debug_steps_per_epoch):
            # stop early during debgging
            break
    eval_vals = []
    N = float(sample_counter.sum)
    for hi, handler in enumerate(stats_handlers):
        stat = handler.sum / float(N)
        logger.info('Stat {} has an avg of {}'.format(hi, stat))
        if hi < n_outputs:
            o_func = output_funcs[hi]
            if o_func is not None:
                stat = o_func(stat)
        if hi >= n_outputs - 1:
            # Note that again n_outputs - 1 is the eval val
            # followed by hallu stats.
            eval_vals.append(stat)
    if input_sess:
        input_sess.close()
    logger.info("evaluation_value={}".format(eval_vals))
    return eval_vals

def load_feature_selection_weights(net_info, model_dir):
    l_omega = []
    for cname in net_info.operable_cell_names:
        names = get_feature_selection_weight_names(net_info[cname], cname)
        info_omegas = read_parameter_val(model_dir, names)
        l_omega.extend(info_omegas)
    return l_omega

def get_l_op_order(net_info, model_dir):
    """
    Helper for child main for computing the operation selection
    order. All of them are reported to parent since parent is
    charge of selection (and may change how many to select
    in the future dynamically)
    """
    l_omega = load_feature_selection_weights(net_info, model_dir)
    l_op_indices = []
    l_op_omega = []
    for omega in l_omega:
        assert len(omega.shape) == 1, 'Omega should be vector'
        abs_omega = np.abs(omega)
        op_indices = list(range(len(omega)))
        op_indices.sort(key=lambda idx: abs_omega[idx], reverse=True)
        op_omega = [ omega[idx] for idx in op_indices]
        l_op_indices.append(op_indices)
        l_op_omega.append(op_omega)
    return l_op_indices, l_op_omega

def feature_selection_cutoff(l_op_indices, l_op_omega, options):
    """
    Helper of server main for selecting the
    feature/operations when we have the statistics.
    """
    l_fs_ops = []
    l_fs_omega = []
    for op_indices, op_omega in zip(l_op_indices, l_op_omega):
        len_op_indices = len(op_indices)
        n_above_threshold = len_op_indices
        if options.hallu_feat_sel_threshold is not None:
            n_above_threshold = len(list(filter(
                lambda val : val > options.hallu_feat_sel_threshold,
                np.abs(op_omega)
            )))
        n_max_fs_ops = len_op_indices
        if options.hallu_feat_sel_num is not None:
            n_max_fs_ops = options.hallu_feat_sel_num
        n_op_indices = min(n_above_threshold, n_max_fs_ops, len_op_indices)
        l_fs_ops.append(op_indices[:n_op_indices])
        l_fs_omega.append(op_omega[:n_op_indices])
    return l_fs_ops, l_fs_omega