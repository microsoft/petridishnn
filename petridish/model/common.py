import numpy as np
import tensorflow as tf

from tensorpack.models import (
    BatchNorm, BNReLU, Conv2D, AvgPooling, MaxPooling,
    LinearWrap, GlobalAvgPooling, FullyConnected, regularize_cost,
    Deconv2D, GroupedConv2D, ResizeImages, SeparableConv2D, Dropout)
from tensorpack.tfutils import argscope
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.tfutils.optimizer import apply_grad_processors
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.callbacks import ClassificationError, ScalarStats
from tensorpack.utils import logger
from tensorpack.utils.argtools import get_data_format

from petridish.info import (
    LayerInfo, LayerInfoList, LayerTypes, CellNetworkInfo
)



def scope_base(layer_idx):
    return "layer{:03d}".format(layer_idx)

def scope_prediction(layer_idx):
    return scope_base(layer_idx) + '.0.pred'

DYNAMIC_WEIGHTS_NAME = 'DYNAMIC_WEIGHTS'

def get_feature_selection_weight_names(layer_info_list, cell_name):
    names = []
    outer_scope = cell_name + '_feature_select/'
    for info in layer_info_list:
        if info.is_candidate and info.merge_op == LayerTypes.MERGE_WITH_WEIGHTED_SUM:
            inner_scope = 'hallu_{}/omega_init:0'.format(info.is_candidate)
            names.append(outer_scope + inner_scope)
    return names

def _get_dim(x, dim):
    """ Get the dimension value of a 4-tensor.
    Helper for _get_C/H/W.
    get_C,H,W will use argscope to set the default dim.
    Otherwise we assume it is NCHW.
    """
    return x.get_shape().as_list()[dim]

def _data_format_to_ch_dim(data_format):
    data_format = get_data_format(data_format, keras_mode=False)
    ch_dim = -1 if data_format == 'NHWC' else 1
    return ch_dim

def feature_to_prediction_and_loss(
        scope_name, l, label, num_classes,
        prediction_feature, ch_dim,
        label_smoothing=0, dense_dropout_keep_prob=1.0, is_last=True):
    """
        Given the feature l at scope_name, compute a classifier.
    """
    with tf.variable_scope(scope_name):
        n_dim = len(l.get_shape().as_list())
        if n_dim == 4 and not is_last:
            with tf.variable_scope('aux_preprocess'):
                l = tf.nn.relu(l)
                l = AvgPooling('pool', l, pool_size=5, strides=3, padding='valid')
                l = Conv2D('conv_proj', l, 128, 1, strides=1, activation=BNReLU)
                shape = l.get_shape().as_list()
                if ch_dim != 1:
                    shape = shape[1:3]
                else:
                    shape = shape[2:4]
                l = Conv2D(
                    'conv_flat', l, 768, shape, strides=1,
                    padding='valid', activation=BNReLU)
                l = tf.layers.flatten(l)
        else:
            l = BNReLU('bnrelu_pred', l)
            ch_in = _get_dim(l, ch_dim)
            if prediction_feature == '1x1':
                ch_out = ch_in
                if n_dim == 4:
                    l = Conv2D('conv1x1', l, ch_out, 1)
                else:
                    assert n_dim == 2, n_dim
                    l = FullyConnected('fc1x1', l, ch_out, activation=tf.identity)
                l = BNReLU('bnrelu1x1', l)

            elif prediction_feature == 'msdense':
                assert n_dim == 2, n_dim
                ch_inter = ch_in
                l = Conv2D('conv1x1_0', l, ch_inter, 3, strides=2)
                l = BNReLU('bnrelu1x1_0', l)
                l = Conv2D('conv1x1_1', l, ch_inter, 3, strides=2)
                l = BNReLU('bnrelu1x1_1', l)

            elif prediction_feature == 'bn':
                l = BatchNorm('bn', l)

            else:
                # Do nothing to the input feature
                pass
            if n_dim > 2:
                l = GlobalAvgPooling('gap', l)

        variables = []
        if num_classes > 0:
            if is_last:
                l = Dropout('drop_pre_fc', l, keep_prob=dense_dropout_keep_prob)
            logits = FullyConnected('linear', l, num_classes, activation=tf.identity)
            variables.append(logits.variables.W)
            variables.append(logits.variables.b)
            tf.nn.softmax(logits, name='preds')
            ## local cost/error_rate
            if label_smoothing > 0:
                one_hot_labels = tf.one_hot(label, num_classes)
                cost = tf.losses.softmax_cross_entropy(\
                    onehot_labels=one_hot_labels, logits=logits,
                    label_smoothing=label_smoothing)
            else:
                cost = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                    logits=logits, labels=label)
            cost = tf.reduce_mean(cost, name='cross_entropy_loss')
            add_moving_summary(cost)
            def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
                return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)),
                   tf.float32, name=name)
            wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
            add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
            wrong5 = prediction_incorrect(logits, label, 5, name='wrong-top5')
            add_moving_summary(tf.reduce_mean(wrong5, name='train-error-top5'))
        else:
            # for regression:
            pred = FullyConnected('linear', l, 1, activation=tf.identity)
            variables.append(pred.variables.W)
            variables.append(pred.variables.b)
            pred = tf.nn.relu(pred)
            tf.identity(pred, name='preds')
            cost = tf.reduce_mean(0.5 * (pred - label) ** 2, name='mean_square_error')
            add_moving_summary(cost)
        return cost, variables

def generate_classification_callbacks(
        layer_info_list,
        name_only=False,
        top1_only=True,
        use_loss=False):
    """
        A list of callbacks for getting validation errors.

        Args:
        layer_info_list
        name_only (bool) :
            decide whether the function only returns a list of names
        top1_only (bool) :
            decide whether the function only returns stats for the top1
        use_loss (bool) :
            decide whether the function returns loss instead of
            error rate. If this is True, top1_only is ignored.
    """
    vcs = []
    names = []
    for info in layer_info_list:
        if info.aux_weight > 0:
            scope_name = scope_prediction(info.id)
            if use_loss:
                tname = scope_name + '/cross_entropy_loss:0'
                vcs.append(ScalarStats(tname, prefix='val_'))
                names.append(tname)
            else:
                tname = scope_name+'/wrong-top1:0'
                vcs.append(ClassificationError(\
                    wrong_tensor_name=tname,
                    summary_name=scope_name+'/val_err'))
                names.append(tname)
                if not top1_only:
                    tname = scope_name+'/wrong-top5:0'
                    vcs.append(ClassificationError(\
                        wrong_tensor_name=tname,
                        summary_name=scope_name+'/val-err5'))
                    names.append(tname)
    if name_only:
        return names
    return vcs

def generate_regression_callbacks(layer_info_list, name_only=False):
    """
        A list of callbacks for getting validation errors.
    """
    vcs = []
    names = []
    for info in layer_info_list:
        if info.aux_weight > 0:
            scope_name = scope_prediction(info.id)
            name = scope_name+'/mean_square_error:0'
            vcs.append(ScalarStats(\
                names=name,
                prefix='val_'))
            names.append(name)
    if name_only:
        return names
    return vcs

def optimizer_common(options):
    lr = _get_lr_variable(options)
    opt = None
    if hasattr(options, 'optimizer'):
        if options.optimizer == 'rmsprop':
            logger.info('RMSPropOptimizer')
            opt = tf.train.RMSPropOptimizer(
                lr, epsilon=options.rmsprop_epsilon)
        elif options.optimizer == 'adam':
            logger.info('AdamOptimizer')
            opt = tf.train.AdamOptimizer(learning_rate=lr)
        elif options.optimizer == 'gd':
            logger.info('GradientDescentOptimizer')
            opt = tf.train.GradientDescentOptimizer(lr)
        elif options.optimizer == 'sgd':
            logger.info('StochasticGradientDescentOptimizer')
            if options.sgd_moment is None:
                logger.info("sgd moment is now specified. Set it to 0")
                options.sgd_moment = 0.0
            opt = tf.train.MomentumOptimizer(lr, options.sgd_moment)
    if opt is None:
        logger.info('(default) MomentumOptimizer')
        opt = tf.train.MomentumOptimizer(lr, options.sgd_moment)
    gradprocs = getattr(options, 'gradprocs', None)
    if gradprocs and isinstance(gradprocs, list) and len(gradprocs) > 0:
        opt = apply_grad_processors(opt, gradprocs)
    return opt

def _get_lr_variable(options):
    assert options.init_lr > 0, options.init_lr
    init_lr = options.init_lr
    lr_decay_method = options.lr_decay_method
    name = 'learning_rate'
    if lr_decay_method is None or lr_decay_method == 'human':
        lr = tf.get_variable(
            name, initializer=float(init_lr),
            trainable=False)

    global_step = get_global_step_var()
    assert options.steps_per_epoch, options.steps_per_epoch
    if lr_decay_method == 'cosine':
        assert options.max_epoch, options.max_epoch
        decay_steps = int(options.steps_per_epoch * options.max_epoch) + 1
        lr = tf.train.cosine_decay(
            init_lr, global_step, decay_steps=decay_steps,
            name=name)

    elif lr_decay_method == 'exponential':
        assert options.lr_decay_every, options.lr_decay_every
        decay_steps = int(options.steps_per_epoch * options.lr_decay_every)
        lr = tf.train.exponential_decay(
            init_lr, global_step,
            decay_steps=decay_steps,
            decay_rate=options.lr_decay,
            staircase=True,
            name=name)

    tf.summary.scalar(name + '-summary', lr)
    return lr

def _type_str_to_type(t_str):
    if t_str == 'int32':
        return tf.int32
    if t_str == 'uint8':
        return tf.uint8
    if t_str == 'float32':
        return tf.float32
    return None
