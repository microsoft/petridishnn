# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, re
import numpy as np
import tensorflow as tf
import struct
rnn = tf.nn.rnn_cell
#cudnn_rnn = tf.contrib.cudnn_rnn
#LSTMCell = cudnn_rnn.CudnnCompatibleLSTMCell #rnn.LSTMCell

from tensorpack.utils import logger
from tensorpack.dataflow import RNGDataFlow, BatchData
from tensorpack.graph_builder import ModelDesc

from tensorpack.models import BatchNorm, BNReLU, Conv2D, AvgPooling, MaxPooling, \
    LinearWrap, GlobalAvgPooling, FullyConnected, regularize_cost, \
    Deconv2D, GroupedConv2D, ResizeImages, SeparableConv2D
from tensorpack.tfutils import argscope, optimizer, summary, gradproc
from tensorpack.tfutils.common import get_tensors_by_names
from tensorpack.tfutils.sessinit import get_model_loader, SaverRestore
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.tower import TowerContext
from tensorpack.tfutils.sesscreate import NewSessionCreator
from tensorpack.utils.serialize import loads, dumps
from tensorpack.models import FullyConnected, Dropout
from tensorpack.train import TrainConfig, DEFAULT_CALLBACKS, SimpleTrainer, launch_train_with_config
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor, OnlinePredictor
from tensorpack.input_source import PlaceholderInput
from tensorpack.callbacks import ModelSaver, InferenceRunner, ScalarStats, \
    ScheduledHyperParamSetter, HumanHyperParamSetter, \
    JSONWriter, ScalarPrinter, TFEventWriter, ProgressBar

from petridish.info import LayerTypes, LayerInfo, LayerInfoList
from petridish.model import residual_bottleneck_layer
from petridish.app.directory import _mi_info_save_fn
from petridish.analysis.old.search_analysis import grep_val_err_from_log



"""
- ModelSearchInfo: info of a model. mi_info can either becomes feat directly through function
critic_feature at test-time or becomes a data_list (a list of feat and target) first in mi_info_to_data

- Test-time:
PredFeature: inputs of xxxCriticDataFlow. see critic_feature. This is used for test-time.

- Training-time Data: inputs for forming DataFlow, which are list of list of value, also contains prediction target.
This the common path for forming xxxCriticDataFlow.

- Training-time xxxCriticDataFlow: expand the Data to suitable prediction feature and target.
This handles xxx specific feature generation.
"""

class ModelSearchInfo(object):

    def __init__(self, mi, pi, sd, fp=None, ve=None, mstr="", stat=[]):
        self.mi = mi
        self.pi = pi
        self.sd = sd
        self.fp = fp
        self.ve = ve
        self.mstr = mstr
        self.stat = stat

    def as_tuple(self):
        return self.mi, self.pi, self.sd, self.fp, self.ve, self.mstr, self.stat

    @staticmethod
    def to_bytes(x):
        return dumps([x.mi, x.pi, x.sd, x.fp, x.ve, x.mstr, x.stat])

    @staticmethod
    def from_bytes(ss):
        x = loads(ss)
        return ModelSearchInfo(*x)

    def to_str(self):
        return str(self.as_tuple())


class CriticTypes(object):
    CONV = 0
    LSTM = 1


def critic_factory(ctrl, is_train, vs_name):
    """
    Generate a critic model
    """
    model = None
    kp = ctrl.controller_dropout_kp if is_train else 1
    lr = ctrl.critic_init_lr if is_train else 0
    if ctrl.critic_type == CriticTypes.CONV:
        model = PetridishConvCritic(max_depth=ctrl.controller_max_depth,
            layer_embedding_size=ctrl.lstm_size,
            dropout_kp=kp, init_lr=lr, vs_name=vs_name, n_hallu_stats=ctrl.n_hallu_stats[vs_name])

    elif ctrl.critic_type == CriticTypes.LSTM:
        model = PetridishLSTMCritic(ctrl.lstm_size, ctrl.num_lstms,
            ctrl.controller_batch_size, ctrl.controller_seq_length,
            kp, lr, vs_name=vs_name, n_hallu_stats=ctrl.n_hallu_stats[vs_name])
    return model


def critic_dataflow_factory(ctrl, data, is_train):
    """
    Generate a critic dataflow
    """
    if ctrl.critic_type == CriticTypes.CONV:
        ds = ConvCriticDataFlow(data, shuffle=is_train, max_depth=ctrl.controller_max_depth)
        ds = BatchData(ds, ctrl.controller_batch_size, remainder=not is_train, use_list=False)
    elif ctrl.critic_type == CriticTypes.LSTM:
        ds = LSTMCriticDataFlow(data, shuffle=is_train)
        ds = BatchData(ds, ctrl.controller_batch_size, remainder=not is_train, use_list=True)
    return ds


"""
CONV based
DataFlow and Critic models

ConvCriticDataFlow
PetridishConvCritic
"""

class ConvCriticDataFlow(RNGDataFlow):

    def __init__(self, data, shuffle=True, max_depth=128):
        super(ConvCriticDataFlow, self).__init__()
        self.shuffle = shuffle
        self._size = N = len(data[0])
        mimgs = np.ones([N, max_depth, max_depth], dtype=int) * LayerTypes.NOT_EXIST
        mflags = np.zeros([N, max_depth, LayerInfoList.n_extra_dim - 1], dtype=int)
        pimgs = np.ones([N, max_depth, max_depth], dtype=int) * LayerTypes.NOT_EXIST
        pflags = np.zeros([N, max_depth, LayerInfoList.n_extra_dim - 1], dtype=int)
        pve = np.asarray(data[2], dtype=float)
        target = data[3]
        stats = data[4:]

        if target is None:
            target = [0.0] * self._size
        for di in range(self._size):
            mseq, pseq = data[0][di], data[1][di]
            mimgs[di], mflags[di] = LayerInfoList.seq_to_img_flag(mseq, max_depth, make_batch=False)
            pimgs[di], pflags[di] = LayerInfoList.seq_to_img_flag(pseq, max_depth, make_batch=False)
        self.dps = [mimgs, mflags, pimgs, pflags, pve, target] + stats

    def size(self):
        return self._size

    def get_data(self):
        idxs = list(range(self._size))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [dp[k]  for dp in self.dps]


class PetridishConvCritic(ModelDesc):

    def __init__(self, max_depth, layer_embedding_size, dropout_kp, init_lr, vs_name, n_hallu_stats=0):
        super(PetridishConvCritic, self).__init__()
        self.max_depth = max_depth
        self.layer_embedding_size = layer_embedding_size
        self.n_flags = LayerInfoList.n_extra_dim - 1
        self.dropout_kp = dropout_kp
        self.init_lr = init_lr
        self.vs_name = vs_name
        self._input_names = ['mimg', 'mflag', 'pimg', 'pflag', 'pve', 'target']
        self._input_names.extend(['hallu_stat_{}'.format(hsi) for hsi in range(n_hallu_stats)])

    def inputs(self):
        input_names = self._input_names
        return [tf.TensorSpec([None, self.max_depth, self.max_depth], tf.int32, input_names[0]),
            tf.TensorSpec([None, self.max_depth, self.n_flags], tf.int32, input_names[1]),
            tf.TensorSpec([None, self.max_depth, self.max_depth], tf.int32, input_names[2]),
            tf.TensorSpec([None, self.max_depth, self.n_flags], tf.int32, input_names[3]),
            tf.TensorSpec([None], tf.float32, input_names[4]),
            tf.TensorSpec([None], tf.float32, input_names[5])] \
            + [ tf.TensorSpec([None], tf.float32, nm) for nm in input_names[6:] ]

    def build_graph(self, *inputs):
        mimg, mflag, pimg, pflag, pve, target = inputs[:6]
        h_stats = list(inputs[6:])

        def img_flag_to_feat(img, flag, embed):
            # B x maxD x maxD x layer_embedding_size
            feat = tf.nn.embedding_lookup(embed, img)
            feat = Dropout(feat, keep_prob=self.dropout_kp)
            # concat connection feature with layer-wise flag feature.
            flag_feat = tf.reshape(tf.tile(flag, [1, 1, self.max_depth]),
                [-1, self.max_depth, self.max_depth, self.n_flags])
            flag_feat = tf.cast(flag_feat, tf.float32)
            l = tf.concat([feat, flag_feat], axis=3, name='concat_feats')

            # feature are now NCHW format
            l = tf.transpose(l, [0, 3, 1, 2])

            # make the feature tensor symmetry on HxW
            lower_l = tf.matrix_band_part(l, -1, 0)
            upper_l = tf.matrix_transpose(lower_l)
            diag_l = tf.matrix_band_part(l, 0, 0)
            l = lower_l + upper_l - diag_l
            return l

        with tf.variable_scope(self.vs_name):
            # embed the connection types.
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            vocab_size = LayerTypes.num_layer_types()
            embeddingW = tf.get_variable('embedding', [vocab_size, self.layer_embedding_size],
                initializer=initializer)
            mfeat = img_flag_to_feat(mimg, mflag, embeddingW)
            pfeat = img_flag_to_feat(pimg, pflag, embeddingW)
            l = tf.concat(values=[mfeat, pfeat], axis=1)

            data_format='channels_first'
            ch_dim = 1
            # network on the combined feature.
            with argscope([Conv2D, Deconv2D, GroupedConv2D, AvgPooling, MaxPooling, \
                    BatchNorm, GlobalAvgPooling, ResizeImages, SeparableConv2D], \
                    data_format=data_format), \
                argscope([Conv2D, Deconv2D, GroupedConv2D, SeparableConv2D], \
                    activation=tf.identity, use_bias=False):

                n_layers_per_scale = 4
                n_scales = 4
                out_filters = l.get_shape().as_list()[ch_dim]
                for si in range(n_scales):
                    for li in range(n_layers_per_scale):
                        name = 'layer{:03d}'.format(si * n_layers_per_scale + li)
                        strides = 1
                        if li == 0 and si > 0:
                            strides = 2
                            out_filters *= 2
                        with tf.variable_scope(name):
                            l = residual_bottleneck_layer('res_btl', l, out_filters, strides, data_format)

                # only use the last output for predicting the child model accuracy
                l = GlobalAvgPooling('gap', l)
                pve = tf.reshape(pve, [-1, 1])
                h_stats = [tf.reshape(hs, [-1, 1]) for hs in h_stats]
                l = tf.concat(values=[pve, l] + h_stats, axis=ch_dim)
                pred = FullyConnected('fully_connect', l, 1, activation=tf.sigmoid)
                pred = tf.reshape(pred, [-1])
                self.pred = tf.identity(pred, name='predicted_accuracy')

                cost = tf.losses.mean_squared_error(target, self.pred)
                self.cost = tf.identity(cost, name='cost')
                add_moving_summary(self.cost)
                return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', self.init_lr, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.AdamOptimizer(lr, beta1=0.0, epsilon=1e-3, use_locking=True)
        return opt


"""
LSTM based
DataFlow and Critic model

LSTMCriticDataFlow
PetridishLSTMCritic
"""

class LSTMCriticDataFlow(RNGDataFlow):

    def __init__(self, data, shuffle=True):
        super(LSTMCriticDataFlow, self).__init__()
        self.shuffle = shuffle
        self._size = len(data[0])
        mseq, pseq, pve, target = data[:4]
        stats = data[4:]
        mlen = list(map(len, mseq))
        plen = list(map(len, pseq))
        self.dps = [mseq, mlen, pseq, plen, pve, target] + stats
        for i in range(len(self.dps)):
            if self.dps[i] is None:
                self.dps[i] = [0.0] * self._size

    def size(self):
        return self._size

    def get_data(self):
        idxs = list(range(self._size))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [dp[k]  for dp in self.dps]


class PetridishLSTMCritic(ModelDesc):

    def __init__(self,
            lstm_size,
            num_lstms,
            batch_size,
            max_seq_len,
            dropout_kp,
            init_lr,
            vs_name,
            n_hallu_stats=0):
        super(PetridishLSTMCritic, self).__init__()
        self.lstm_size = lstm_size
        self.num_lstms = num_lstms
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dropout_kp = dropout_kp
        self.init_lr = init_lr
        self.vs_name = vs_name
        self._input_names = ['mseq', 'mlen', 'pseq', 'plen', 'pve', 'target']
        self._input_names.extend(['hallu_stat_{}'.format(hsi) for hsi in range(n_hallu_stats)])


    def inputs(self):
        input_names = self._input_names
        return [tf.TensorSpec([self.batch_size, self.max_seq_len], tf.int32, input_names[0]),
            tf.TensorSpec([self.batch_size], tf.int32, input_names[1]),
            tf.TensorSpec([self.batch_size, self.max_seq_len], tf.int32, input_names[2]),
            tf.TensorSpec([self.batch_size], tf.int32, input_names[3]),
            tf.TensorSpec([self.batch_size], tf.float32, input_names[4]),
            tf.TensorSpec([self.batch_size], tf.float32, input_names[5])] \
            + [ tf.TensorSpec([None], tf.float32, nm) for nm in input_names[6:] ]

    def build_graph(self, *inputs):
        mseq, mlen, pseq, plen, pve, target = inputs[:6]
        h_stats = list(inputs[6:])

        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope(self.vs_name):
            # Feature embedding
            vocab_size = LayerTypes.num_layer_types()
            embeddingW = tf.get_variable('embedding', [vocab_size, self.lstm_size],
                initializer=initializer)
            mfeat = tf.nn.embedding_lookup(embeddingW, mseq)  # B x seqlen x hiddensize
            mfeat = Dropout(mfeat, keep_prob=self.dropout_kp)
            pfeat = tf.nn.embedding_lookup(embeddingW, pseq)
            pfeat = Dropout(pfeat, keep_prob=self.dropout_kp)

            # LSTM structures
            def get_basic_cell():
                cell = rnn.LSTMCell(num_units=self.lstm_size, initializer=initializer,
                    reuse=tf.get_variable_scope().reuse)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                return cell
            cells = rnn.MultiRNNCell([get_basic_cell() for _ in range(self.num_lstms)])
            #cells =cudnn_rnn.CudnnLSTM(self.num_lstms, self.lstm_size, dropout=1 - self.dropout_kp,
            #    kernel_initializer=initializer)
            # initial state
            mstate = cells.zero_state(self.batch_size, dtype=tf.float32)
            pstate = cells.zero_state(self.batch_size, dtype=tf.float32)

            # apply LSTMs on the feature embedding
            with tf.variable_scope('LSTM'):
                mout, mstate = tf.nn.dynamic_rnn(cells, mfeat,
                    initial_state=mstate,
                    sequence_length=mlen)
                pout, pstate = tf.nn.dynamic_rnn(cells, pfeat,
                    initial_state=pstate,
                    sequence_length=plen)

            # only use the last output for predicting the child model accuracy
            mlen = tf.cast(tf.reshape(mlen, [self.batch_size, 1]), dtype=tf.float32)
            plen = tf.cast(tf.reshape(plen, [self.batch_size, 1]), dtype=tf.float32)
            pve = tf.reshape(pve, [self.batch_size, 1])
            h_stats = [tf.reshape(hs, [-1, 1]) for hs in h_stats]
            feat = tf.concat(values=[mout[:,-1], pout[:,-1], pve] + h_stats, axis=1)
            pred = FullyConnected('fully_connect', feat, 1, activation=tf.sigmoid)
            pred = tf.reshape(pred, [self.batch_size])
            self.pred = tf.identity(pred, name='predicted_accuracy')

            cost = tf.losses.mean_squared_error(target, self.pred)
            self.cost = tf.identity(cost, name='cost')
            add_moving_summary(self.cost)
            return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', self.init_lr, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.AdamOptimizer(lr, beta1=0.0, epsilon=1e-3, use_locking=True)
        return opt


"""
Prediction using critic models
"""

def critic_predict_dataflow(ctrl, data, log_dir, model_dir, vs_name):
    """
    Prediction on a dataflow, used for testing a large batch of data
    """
    ckpt = tf.train.latest_checkpoint(model_dir)
    if not ckpt:
        outputs = [0] * len(data[0])
        logger.info("No model exists. Do not sort")
        return outputs
    model = critic_factory(ctrl, is_train=False, vs_name=vs_name)
    ds_val = critic_dataflow_factory(ctrl, data, is_train=False)
    output_names = ['{}/predicted_accuracy:0'.format(vs_name)]

    session_config=None
    if ctrl.critic_type == CriticTypes.LSTM:
        session_config = tf.ConfigProto(device_count = {'GPU': 0})
    pred_config = PredictConfig(
        model=model,
        input_names=model.input_names,
        output_names=output_names,
        session_creator=NewSessionCreator(config=session_config),
        session_init=SaverRestore(ckpt)
    )

    #with tf.Graph().as_default():
    predictor = SimpleDatasetPredictor(pred_config, ds_val)
    outputs = []
    for o in predictor.get_result():
        outputs.extend(o[0])
    return outputs


class OfflinePredictorWithSaver(OnlinePredictor):
    """ A predictor built from a given config.
        A single-tower model will be built without any prefix.
        A saver will be created before session creation for updating the params."""

    def __init__(self, config):
        """
        Args:
            config (PredictConfig): the config to use.
        """
        self._input_names = config.input_names
        self.graph = config._maybe_create_graph()
        with self.graph.as_default():
            input = PlaceholderInput()
            input.setup(config.input_signature)
            with TowerContext('', is_training=False):
                config.tower_func(*input.get_input_tensors())

            input_tensors = get_tensors_by_names(config.input_names)
            output_tensors = get_tensors_by_names(config.output_names)

            config.session_init._setup_graph()
            self.saver = tf.train.Saver()
            init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
            self.sess = config.session_creator.create_session()
            self.sess.run(init_op)
            config.session_init._run_init(self.sess)
            super(OfflinePredictorWithSaver, self).__init__(
                input_tensors, output_tensors, config.return_input, self.sess)

    def restore(self, ckpt):
        self.saver.restore(self.sess, ckpt)

    def input_index(self, input_name):
        try:
            return self._input_names.index(input_name)
        except ValueError:
            return None



def critic_predictor(ctrl, model_dir, vs_name):
    """
    Create an OfflinePredictorWithSaver for test-time use.
    """
    model = critic_factory(ctrl, is_train=False, vs_name=vs_name)
    output_names = ['{}/predicted_accuracy:0'.format(vs_name)]
    session_config=None
    if ctrl.critic_type == CriticTypes.LSTM:
        session_config = tf.ConfigProto(device_count = {'GPU': 0})
    pred_config = PredictConfig(
        model=model,
        input_names=model.input_names,
        output_names=output_names,
        session_creator=NewSessionCreator(config=session_config)
    )
    if model_dir:
        ckpt = tf.train.latest_checkpoint(model_dir)
        logger.info("Loading {} predictor from {}".format(vs_name, ckpt))
        if ckpt:
            pred_config.session_init = SaverRestore(ckpt)
    predictor = OfflinePredictorWithSaver(pred_config)
    return predictor


def critic_feature(ctrl, mi_info, l_mi, l_lil=None):
    """
    Compute from mi_info to the feature for prediction, the prediction_target is not computed.
    """
    if isinstance(l_mi, int):
        l_mi = [l_mi]
        l_lil = [l_lil]
    if l_lil is None:
        l_lil = [None] * len(l_mi)
    feats = None
    for mi, lil in zip(l_mi, l_lil):
        if isinstance(mi_info, list):
            assert len(mi_info) > mi
        elif isinstance(mi_info, dict):
            assert mi in mi_info.keys()

        info = mi_info[mi]
        mi, pi, sd, fp, ve, mstr, stat = info.as_tuple()
        ve = 0.0 if ve is None or ve > 1.0 else ve
        if lil:
            mseq = LayerInfoList.to_seq(lil)
        else:
            mseq = LayerInfoList.str_to_seq(mstr)

        pinfo = mi_info[pi]
        pseq = LayerInfoList.str_to_seq(pinfo.mstr)
        pve = pinfo.ve

        if ctrl.critic_type == CriticTypes.CONV:
            mimg, mflag = LayerInfoList.seq_to_img_flag(mseq, ctrl.controller_max_depth, make_batch=True)
            pimg, pflag = LayerInfoList.seq_to_img_flag(pseq, ctrl.controller_max_depth, make_batch=True)
            feat = [ mimg, mflag, pimg, pflag, pve, ve ]
        elif ctrl.critic_type == CriticTypes.LSTM:
            feat = [ mseq, len(mseq), pseq, len(pseq), pve, ve ]
        feat.extend(stat)
        if feats is None:
            feats = [ [val] for val in feat ]
        else:
            for col, x in zip(feats, feat):
                col.append(x)
    return feats


"""
Forming data and training critic models
"""

def mi_info_to_data(mi_info, use_min_subtree):
    """
    Compute data for critic dataflow. The seq are not expanded to the feature,
    as they will be down by dataflow.
    """
    dps = None
    mi_info_keys = mi_info.keys() if isinstance(mi_info, dict) else range(len(mi_info))
    if not use_min_subtree:
        for mi in mi_info_keys:
            mi, pi, sd, fp, ve, mstr, stat = mi_info[mi].as_tuple()
            if ve is None or ve > 1:
                continue
            if sd % 2 == 1:
                continue
            pinfo = mi_info[pi]
            pstr = pinfo.mstr
            pve = pinfo.ve
            vals = [LayerInfoList.str_to_seq(mstr),
                LayerInfoList.str_to_seq(pstr),
                pve,
                ve] + stat
            if dps is None:
                dps = [[] for _ in vals]
            for dp, val in zip(dps, vals):
                dp.append(val)

    else:
        mi_min_child = dict()
        for mi in mi_info_keys:
            mi, pi, sd, fp, ve, mstr, stat = mi_info[mi].as_tuple()
            if pi < 0 or mi == pi:
                continue
            if ve is None or ve > 1:
                continue
            if sd % 2 == 0:
                old = mi_min_child.get(pi, 2.0)
                mi_min_child[pi] = min(old, ve)
        for mi in mi_min_child:
            min_ve = mi_min_child[mi]
            mi, pi, sd, fp, ve, mstr, stat = mi_info[mi].as_tuple()
            pinfo = mi_info[pi]
            pstr = pinfo.mstr
            pve = pinfo.ve
            vals = [LayerInfoList.str_to_seq(mstr),
                LayerInfoList.str_to_seq(pstr),
                pve,
                min_ve] + stat
            if dps is None:
                dps = [[] for _ in vals]
            for dp, val in zip(dps, vals):
                dp.append(val)
    return dps


def critic_train(ctrl, data, log_dir, model_dir, prev_dir, vs_name, split_train_val=False):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    lr_schedule = []
    max_epoch = ctrl.critic_train_epoch
    lr = ctrl.critic_init_lr
    for epoch in range(0, max_epoch):
        if epoch % 1 == 0:
            lr_schedule.append((epoch+1, lr))
            lr *= 0.9
    ds_size = len(data[0])
    idxs = list(range(ds_size))
    np.random.shuffle(idxs)

    if split_train_val:
        train_size = ds_size * 9 // 10
        if train_size == 0:
            train_size = ds_size
        val_start = train_size
    else:
        train_size = ds_size
        val_start = ds_size * 9 // 10
    if ds_size - val_start == 0:
        val_start = 0

    data_train = [ [col[k] for k in idxs[:train_size]] for col in data ]
    data_val = [ [col[k] for k in idxs[val_start:]] for col in data ]

    model = critic_factory(ctrl, is_train=True, vs_name=vs_name)
    ds_train = critic_dataflow_factory(ctrl, data_train, is_train=True)
    ds_val = critic_dataflow_factory(ctrl, data_val, is_train=False)
    session_config = None
    device = 0
    if ctrl.critic_type == CriticTypes.LSTM:
        session_config = tf.ConfigProto(device_count = {'GPU': 0})
        device = -1
    extra_callbacks = DEFAULT_CALLBACKS()
    extra_callbacks = list(filter(lambda x : not isinstance(x, ProgressBar), extra_callbacks))
    logger.info("Extra callbacks are {}".format(list(map(lambda x : x.__class__, extra_callbacks))))
    # Put this into callbacks for in-training validation/inferencing
    inference_callback = InferenceRunner(ds_val,
        [ScalarStats('{}/cost'.format(vs_name))], device=device)
    config = TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=model_dir, max_to_keep=1,
                keep_checkpoint_every_n_hours=100),
            ScheduledHyperParamSetter('learning_rate', lr_schedule)
        ],
        extra_callbacks=extra_callbacks,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()], #, TFEventWriter()],
        steps_per_epoch=ds_train.size(),
        max_epoch=max_epoch,
        session_config=session_config
    )
    ckpt = tf.train.latest_checkpoint(prev_dir if prev_dir else model_dir)
    if ckpt:
        config.session_init = SaverRestore(ckpt)
    launch_train_with_config(config, SimpleTrainer())


def crawl_data_and_critic_train(controller, data_dir, crawl_dirs, log_dir,
                                model_dir, prev_dir, vs_name, store_data=False, split_train_val=False):
    if ',' in crawl_dirs:
        crawl_dirs = crawl_dirs.strip().split(',')
    data = crawl_critic_data(data_dir, crawl_dirs,
                critic_name=vs_name, subtree=(vs_name == 'q_hallu'),
                overwrite=store_data, use_remote_logs=False)
    # this only happens on remote, which has options.queue_name set by train_critic_remotely
    vs_name = "" if vs_name is None else vs_name
    critic_train(controller, data, log_dir, model_dir, prev_dir, vs_name, split_train_val)


def crawl_critic_data(data_dir, crawl_dirs, critic_name,
        subtree=False, overwrite=False, use_remote_logs=False):
    """
    Args:
    data_dir (str): dir that contains the original data npz.
    crawl_dirs (list/str) : either a list of log_dir_roots of runs,
            or a comma separated list of the log_dir_roots
    critic_name (str) : use for determing which type of critic it is.
        For loading/save data.
    subtree (bool) : Whether the target prediction is based on performance
        of subtree of the root. (TODO expand this to be based on depth)
    overwrite (bool) : Whether to over write the data npz after crawling.
    use_remote_log (bool) : whether use remote jobs' log to update ve.
        Used by old logs that do not log remote ve in main log.
    """
    data_bin = os.path.join(data_dir, '{}.bin'.format(critic_name))

    dps = None

    if os.path.exists(data_bin):
        with open(data_bin, 'rb') as fin:
            ss = fin.read()
            dps = loads(ss)

    if crawl_dirs:
        if isinstance(crawl_dirs, str):
            crawl_dirs = crawl_dirs.strip().split(',')
        for cdn in crawl_dirs:
            dn_list = os.listdir(cdn)
            if len(dn_list) == 0:
                continue
            if 'log.log' not in dn_list:
                dn = os.path.join(cdn, dn_list[0])
            else:
                dn = cdn

            mi_info = np.load(_mi_info_save_fn(dn), encoding='bytes')['mi_info']
            new_dps = mi_info_to_data(mi_info, subtree)
            if dps is None:
                dps = new_dps
            else:
                for dp, new_dp in zip(dps, new_dps):
                    dp.extend(new_dp)

    if overwrite:
        try:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            with open(data_bin, 'wb') as fout:
                fout.write(dumps(dps))
        except:
            logger.info("Failed to save data in {}".format(data_bin))

    n_samples = 0 if dps is None else len(dps[0])
    logger.info('Critic data crawler for {} found n_samples={}'.format(critic_name, n_samples))
    return dps


def crawl_critic_data_from_server_log(log_fn):
    """
    deprecated do not use
    """
    # info is a list, 0 : mi, 1 : pi, 2 : sd, 3 : ve, 4: mstr
    mi_info = dict()
    is_grep_mstr = False
    mi = 0
    with open(log_fn, 'rt') as fin:
        for line in fin:
            line = line.strip()
            reret = re.search(r'LayerInfoList -H is', line)
            if reret:
                is_grep_mstr = True
                mstr = []
                continue
            if is_grep_mstr:
                reret = re.search(r'^{.*}$', line)
                if reret:
                    mstr.append(reret.group(0))
                    continue
                else:
                    mstr = LayerInfoList.DELIM.join(mstr)
                    mi_info[mi].mstr = mstr
                    is_grep_mstr = False

            reret = re.search(r'mi=([0-9]*) pi=([0-9]*) sd=([0-9]*)', line)
            if reret:
                mi = int(reret.group(1))
                pi = int(reret.group(2))
                sd = int(reret.group(3))
                mi_info[mi] = ModelSearchInfo(mi, pi, sd, None, None)
                continue

            reret = re.search(r'mi=([0-9]*) val_err=([0-9\.]*)$', line)
            if reret:
                mi = int(reret.group(1))
                ve = float(reret.group(2))
                mi_info[mi].ve = ve
    return mi_info


def crawl_ve_from_remote_logs(mi_info, dn):
    """
    deprecated do not use

    Args:
    mi_info : a dict mapping from model iter to ModelSearchInfo
    dn : directory path of the one that directly contains the server log.log, i.e.,
    the remote logs are in {dn}/{model_iter}/log.log
    """
    for mi in mi_info:
        info = mi_info[mi]
        if info.ve is None or info.ve > 1.0:
            log_fn = os.path.join(dn, str(mi), 'log.log')
            if os.path.exists(log_fn):
                ve = grep_val_err_from_log(log_fn)
                mi_info[mi].ve = ve
    return mi_info



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl_data', default=False, action='store_true')
    parser.add_argument('--crawl_data_and_critic_train', default=False, action='store_true')
    parser.add_argument('--critic_predict_dataflow', default=False, action='store_true')
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--crawl_dirs', type=str, default="")
    parser.add_argument('--vs_name', type=str, default='q_child')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--use_remote_logs', default=False, action='store_true')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--prev_dir', type=str, default=None)

    args = parser.parse_args()
    ctrl = lambda : 'meow'
    ctrl.critic_type = CriticTypes.LSTM
    ctrl.controller_max_depth = 128
    ctrl.controller_seq_length = None
    ctrl.controller_batch_size = 1 if ctrl.critic_type == CriticTypes.LSTM else 5
    ctrl.controller_dropout_kp = 0.5
    ctrl.lstm_size = 32
    ctrl.num_lstms = 2
    ctrl.critic_init_lr = 1e-3 if ctrl.critic_type == CriticTypes.LSTM else 1e-3
    ctrl.critic_train_epoch = 40
    ctrl.queue_names = ['q_parent', 'q_hallu', 'q_child']
    ctrl.n_hallu_stats = { 'q_parent' : 0, 'q_hallu' : 0, 'q_child' : 1 }

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logger.set_logger_dir(args.log_dir, action='d')

    """
    Sample command:
    """

    if args.crawl_data:
        data = crawl_critic_data(data_dir=args.data_dir,
            crawl_dirs=args.crawl_dirs,
            critic_name=args.vs_name, subtree=args.vs_name == 'q_hallu',
            overwrite=args.overwrite, use_remote_logs=args.use_remote_logs)
    elif args.crawl_data_and_critic_train:
        crawl_data_and_critic_train(ctrl, args.data_dir, args.crawl_dirs,
            args.log_dir, args.model_dir, args.prev_dir, args.vs_name, args.overwrite,
            split_train_val=True)
    elif args.critic_predict_dataflow:
        d_npz = np.load(os.path.join(args.data_dir, '{}.npz'.format(args.vs_name)), encoding='bytes')
        data = [list(d_npz[key]) for key in ['mseq', 'pseq', 'pve', 'target'] ]
        yp = critic_predict_dataflow(ctrl, data, args.log_dir, args.model_dir, args.vs_name)
