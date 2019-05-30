import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
import struct 
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from tensorpack import *
from tensorpack.utils import logger, utils, fs, stats

from anytime_models.examples import get_augmented_data

ILSVRC_DEFAULT_BATCH_SIZE = 256
ILSVRC_DEFAULT_INPUT_SIZE = 224

def log_init(args, model_cls):
    """
        Set the log root according to the args.log_dir and 
        log run info
    """
    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir(action='k')
    logger.info("Arguments: {}".format(args))
    logger.info("Model class is {}".format(model_cls))
    logger.info("TF version: {}".format(tf.__version__))

def grep_starting_epoch(load, steps_per_epoch):
    """
    load : the checkpoint to load from 
    steps_per_epoch : number of batches per epoch

    return:
    starting_epoch : the starting epoch number for the main_loop
    """
    starting_epoch = 1
    if load:
        dir_name, ckpt = os.path.split(load)
        logger.info("{} exists for loading".format(load))
        if ckpt != "checkpoint":
            file_names = [ckpt]
        else:
            file_names = os.listdir(dir_name)
        logger.info("The files we are checking are {}".format(file_names))
        max_step = 0
        for fn in file_names:
            name, ext = os.path.splitext(fn)
            if name[:5] == 'model':
                try:
                    step = int(name[name.rfind('-')+1:])
                    max_step = max(max_step, step)
                    logger.info("{} is at step {}".format(fn, step)) 
                except:
                    continue
        starting_epoch = 1 + max_step / steps_per_epoch
    return starting_epoch


def grep_init_lr(starting_epoch, lr_schedule):
    """
    starting_epoch : starting epoch index (1 based)
    lr_schedule : list of [(epoch, val), ...]. It is assumed to be sorted 
        by epoch

    return
    init_lr : learning rate at the starting epoch
    """
    init_lr = lr_schedule[0][1]
    for e, v in lr_schedule:
        if starting_epoch > e:
            init_lr = v
        else:
            break
    return init_lr 


def parser_add_app_arguments(parser):
    parser.add_argument('--ds_name', help='name of dataset',
                        type=str, default='ilsvrc')
    parser.add_argument('--data_dir', help='ILSVRC dataset dir that contains the tf records directly',
                        default=os.getenv('PT_DATA_DIR', './data'))
    parser.add_argument('--log_dir', help='log_dir for stdout',
                        default=os.getenv('PT_OUTPUT_DIR', './train_logs'))
    parser.add_argument('--model_dir', help='dir for saving models',
                        default=os.getenv('PT_OUTPUT_DIR', './train_logs'))
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--evaluate', help='a comma separated list containing [train, test, val]',
                        default="", type=str)
    parser.add_argument('--do_validation', help='whether to use validation set',
                        default=False, action='store_true')
    parser.add_argument('--num_anytime_preds', help='Number of anytime predictions',
                        default=None, type=int)

    # For alter label and ann_policy, both of which don't really work
    #parser.add_argument('--store_final_prediction', help='wheter evaluation stores final prediction',
    #                    default=False, action='store_true')
    #parser.add_argument('--store_basename', help='basename_<train/test>.bin for storing the logits',
    #                    type=str, default='distill_target')
    #parser.add_argument('--store_feats_preds', help='whether store final feature and predictins in npz', 
    #                    default=False, action='store_true')
    #parser.add_argument('--store_images_labels', help='whether store input image and labels in npz during eval', 
    #                    default=False, action='store_true')
    return parser


def cosine_learning_rate(lr_max=0.05, lr_min=0.001, initial_period=10, period_multiplier=2, max_epoch=300):
    lr_schedule = []
    curr_step = 0
    curr_period = initial_period
    for i in range(max_epoch):
        if i == max_epoch - 1:
            lr = lr_min
        else:
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * curr_step / (curr_period - 1)))
        curr_step += 1
        if curr_step == curr_period:
            curr_step = 0
            curr_period *= period_multiplier
            if curr_period + i + 1 > max_epoch:
                curr_step = curr_period - (max_epoch - i - 1)
        lr_schedule.append((i, lr))
    return lr_schedule



def train_or_test_ilsvrc(args, model_cls):
    # Fix some common args like num_classes 
    args = ilsvrc_fix_args(args)
    log_init(args, model_cls)

    # If test 
    args.evaluate = list(filter(bool, args.evaluate.split(',')))
    do_eval = len(args.evaluate) > 0
    if do_eval:
        for subset in args.evaluate:
            if subset in ['train', 'val']:
                evaluate_ilsvrc(args, subset, model_cls)
        return 

    # else train
    lr_schedule, max_epoch = ilsvrc_lr_schedule(args.batch_size, lr_option='resnet')
    config = ilsvrc_train_config(args, model_cls, lr_schedule, max_epoch)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(args.nr_gpu))


def evaluate_ilsvrc(args, subset, model_cls):
    ds = get_augmented_data.get_ilsvrc_augmented_data(subset, args, do_multiprocess=False)
    INPUT_SIZE = ILSVRC_DEFAULT_INPUT_SIZE
    model = model_cls(INPUT_SIZE, args)

    args.store_basename = None # This is disabled for now; it used to help storing predictions

    output_names = []
    accs = []
    n_preds = 0
    if args.num_anytime_preds == 0:
        output_names.append('dummy_image_mean')
    else:
        for i, w in enumerate(model.weights):
            if w > 0:
                n_preds += 1
                scope_name = model.compute_scope_basename(i)
                scope_name = model.prediction_scope(scope_name) 
                output_names.append('{}/wrong-top1'.format(scope_name))
                output_names.append('{}/wrong-top5'.format(scope_name))
                accs.extend([stats.RatioCounter(), stats.RatioCounter()])
                #output_names.append('{}/linear/output:0'.format(scope_name))
            if args.num_anytime_preds > 0 and n_preds >= args.num_anytime_preds:
                break

    pred_config = PredictConfig(
        model=model,
        input_names=['input', 'label'],
        output_names=output_names
    )
    if args.load:
        pred_config.session_init = get_model_loader(args.load)
    pred = SimpleDatasetPredictor(pred_config, ds)

    if args.store_basename is not None:
        store_fn = args.store_basename + "_{}.bin".format(subset)
        f_store_out = open(store_fn, 'wb')

    n_batches = 0
    import time
    start_time = time.time() 
    for o in pred.get_result():
        n_batches += 1
        if args.num_anytime_preds == 0:
            continue
        if args.store_basename is not None:
            preds = o[0]
            f_store_out.write(preds)
        batch_size = o[0].shape[0] 
        for i, acc in enumerate(accs):
            acc.feed(o[i].sum(), batch_size)
    logger.info('Inference finished, time: {:.4f}sec'.format(time.time() - start_time))
    if args.num_anytime_preds != 0:
        for i, name in enumerate(output_names):
            logger.info("Name {}, RatioCount {}".format(name, accs[i].ratio))

    if args.store_basename is not None:
        f_store_out.close()


def ilsvrc_fix_args(args):
    """
        Update the args with fixed parameter in ilsvrc
    """
    args.ds_name="ilsvrc"
    args.num_classes == 1000
    # GPU will handle mean std transformation to save CPU-GPU communication
    args.do_mean_std_gpu_process = True
    args.input_type = 'uint8'
    args.mean = get_augmented_data.ilsvrc_mean
    args.std = get_augmented_data.ilsvrc_std
    #assert args.do_mean_std_gpu_process and args.input_type == 'uint8'
    #assert args.mean is not None and args.std is not None

    decay_power = args.batch_size / float(ILSVRC_DEFAULT_BATCH_SIZE) 
    args.batch_norm_decay=0.9**decay_power  # according to Torch blog
    return args


def ilsvrc_lr_schedule(batch_size, lr_option='resnet'):
    """
        lr_option : option for the learning rate schedule
            'resnet' - the original kaiming schedule
            'resnet-v2' - an alternative that is adapted from CIFAR/SVHN lr.
    """
    # Learning rate schedule
    decay_power = batch_size / float(ILSVRC_DEFAULT_BATCH_SIZE) 
    max_epoch = 128
    if lr_option == 'resnet':
        lr_base = 0.1 * decay_power
        lr_schedule = [(1, lr_base), (30, 1e-1 * lr_base), 
            (60, 1e-2 * lr_base), (90, 1e-3 * lr_base),
            (105, 1e-4 * lr_base) ] 

    elif lr_option == 'resnet-v2':
        lr_base = 0.05 * decay_power
        lr_schedule = [(1, lr_base), (60, 1e-1 * lr_base),
            (90, 1e-2 * lr_base), (105, 1e-3 * lr_base)]

    else:
        raise Exception("Unknown learning rate schedule option {}".format(lr_option))
    return lr_schedule, max_epoch


def ilsvrc_train_config(args, model_cls, lr_schedule, max_epoch):
    """
        Get a TrainConfig for training on ilsvrc.

        args : input argument from parser i.e. options
        model_cls : name of the class of the network, e.g., AnytimeResNet
        lr_schedule : [(epoch, lr_val),...]
        max_epoch : maximum epochs to run

        return

        a TrainConfig instance
    """
    # Datasets for train and validation
    get_data = get_augmented_data.get_ilsvrc_augmented_data
    dataset_train = get_data('train', args, do_multiprocess=True)
    dataset_val = get_data('val', args, do_multiprocess=True) 

    # initial lr and epoch
    steps_per_epoch = dataset_train.size() // args.nr_gpu
    starting_epoch = grep_starting_epoch(args.load, steps_per_epoch)
    logger.info("The starting epoch is {}".format(starting_epoch))
    args.init_lr = grep_init_lr(starting_epoch, lr_schedule)
    logger.info("The starting learning rate is {}".format(args.init_lr))

    # Model, call backs
    INPUT_SIZE = ILSVRC_DEFAULT_INPUT_SIZE
    model = model_cls(INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, max_to_keep=2, keep_checkpoint_every_n_hours=100),
            InferenceRunner(dataset_val, classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate'),
        ] + loss_select_cbs,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        starting_epoch=starting_epoch,
        max_epoch=max_epoch,
    )


#################
# CIFAR/SVHN
#################

def cifar_svhn_train_or_test(args, model_cls):
    """
        args : parsed arguments 
        model_cls : class name of the model to run

        return :
    """
    log_init(args, model_cls)

    # generate a list of none-empty strings for specifying the splits
    args.evaluate = list(filter(bool, args.evaluate.split(',')))
    do_eval = len(args.evaluate) > 0
    evaluate = evaluate_cifar_svhn

    ## Set dataset-network specific assert/info
    if args.ds_name == 'cifar10' or args.ds_name == 'cifar100':
        if args.ds_name == 'cifar10':
            args.num_classes = 10
        else:
            args.num_classes = 100
        args.regularize_coef = 'decay'
        INPUT_SIZE = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_augmented_data.get_cifar_augmented_data
        ds_train = get_data('train', args, do_multiprocess=not do_eval, 
            do_validation=args.do_validation)
        ds_val = get_data('test', args, 
            do_multiprocess=False, do_validation=args.do_validation)

        lr_schedule = \
            [(1, 0.1), (140, 0.01), (210, 0.001), (250, 0.0002)]
        max_epoch = 300

        if do_eval:
            for eval_name in args.evaluate:
                if eval_name == 'train':
                    ds = ds_train
                elif eval_name == 'test':
                    ds = ds_val
                evaluate(model_cls, ds, eval_name)
            return

    elif args.ds_name == 'svhn':
        args.num_classes = 10
        args.regularize_coef = 'decay'
        INPUT_SIZE = 32
        fs.set_dataset_path(path=args.data_dir, auto_download=False)
        get_data = get_augmented_data.get_svhn_augmented_data
        
        if do_eval:
            if 'train' in args.evaluate:
                args.evaluate.append('extra')
            for eval_name in args.evaluate:
                ds = get_data(eval_name, args, do_multiprocess=False)
                evaluate(model_cls, ds, eval_name)
            return

        ## Training model 
        ds_train = get_data('train', args, do_multiprocess=True)
        ds_val = get_data('test', args, do_multiprocess=False)

        lr_schedule = \
            [(1, 0.1), (15, 0.01), (30, 0.001), (45, 0.0002)]
        max_epoch = 60


    # svhn/cifar are small enough so that we restart from scratch if interrupted.
    steps_per_epoch = ds_train.size() // args.nr_gpu
    starting_epoch = grep_starting_epoch(args.load, steps_per_epoch)
    logger.info("The starting epoch is {}".format(starting_epoch))
    args.init_lr = grep_init_lr(starting_epoch, lr_schedule)
    logger.info("The starting learning rate is {}".format(args.init_lr))

    model = model_cls(INPUT_SIZE, args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    config = TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, max_to_keep=2, keep_checkpoint_every_n_hours=100),
            InferenceRunner(ds_val,
                            [ScalarStats('cost')] + classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ] + loss_select_cbs,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
        starting_epoch=starting_epoch
    )
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(args.nr_gpu))


def evaluate_cifar_svhn(model_cls, ds, eval_names):
    assert args is not None, args
    model = model_cls(INPUT_SIZE, args) 

    output_names = []
    feature_names = []
    for i, w in enumerate(model.weights):
        if w > 0:
            output_names.append('layer{:03d}.0.pred/linear/output:0'.format(i))
            feature_names.append('layer{:03d}.0.end/add:0'.format(i))
    n_outputs = len(output_names)

    pred_config = PredictConfig(
        model=model,
        input_names=['input', 'label'],
        output_names=['input', 'label'] + output_names[-1:] + feature_names[-1:])
    if args.load:
        pred_config.session_init = SaverRestore(args.load)
    
    pred = SimpleDatasetPredictor(pred_config, ds)

    if args.store_final_prediction:
        store_fn = args.store_basename + "_{}.bin".format(eval_name)
        f_store_out = open(store_fn, 'wb')

    l_labels = []
    l_images = []
    l_preds = []
    l_feats = []
    for idx, output in enumerate(pred.get_result()):
        # o contains a list of predictios at various locations; each pred contains a small batch
        image, label = output[0:2]
        l_labels.extend(label)
        l_images.extend(image)
        anytime_preds = output[2]
        anytime_feats = output[3]

        l_preds.extend(anytime_preds)
        l_feats.extend(anytime_feats)
        
        if args.store_final_prediction:
            preds = anytime_preds
            f_store_out.write(preds)

    # since the labels comes in batches
    l_labels = np.asarray(l_labels)
    l_images = np.asarray(l_images)
    l_preds = np.asarray(l_preds)
    l_feats = np.asarray(l_feats)
    if args.store_feats_preds:
        if args.store_images_labels:
            np.savez(args.store_basename + '_XY.npz', l_images=l_images, l_labels=l_labels)
        np.savez(args.store_basename + '.npz', l_preds=l_preds, l_feats=l_feats)

    logger.info("N samples predicted: {}".format(len(l_labels)))
    if args.store_final_prediction:
        f_store_out.close()
        
        ## report accuracy of the stored predicitons
        with open(store_fn, 'rb') as fin:
            n_wrong = 0
            row_len = 4 * args.num_classes
            for label in l_labels:
                contents = fin.read(row_len)
                logit = np.asarray(struct.unpack('f'*args.num_classes, contents)).reshape([1, args.num_classes])
                n_wrong += int(np.argmax(logit, axis=1) != label)

        error_rate = n_wrong / np.float32(len(l_labels))
        logger.info("Verify error rate of the stored prediction to be {}".format(error_rate))
