# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import tensorflow as tf

from tensorpack.graph_builder import ModelDesc
from tensorpack.models import (
    BatchNorm, BNReLU, Conv2D, AvgPooling, MaxPooling,
    LinearWrap, GlobalAvgPooling, FullyConnected, regularize_cost,
    Deconv2D, GroupedConv2D, ResizeImages, SeparableConv2D, Dropout)
from tensorpack.tfutils import argscope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils import logger

from petridish.info import (
    LayerInfo, LayerInfoList, LayerTypes, CellNetworkInfo
)
from petridish.model.hallu import (
    _init_hallu_record, _update_hallu_record, _hallu_stats_graph,
    _hallu_stats_graph_merged, get_hallu_stats_output_names
)
from petridish.model.common import (
    optimizer_common, _type_str_to_type, feature_to_prediction_and_loss
)
from petridish.model.layer import (
   construct_layer, candidate_gated_layer, _init_feature_select
)

rnn = tf.contrib.rnn
RNNCell = tf.nn.rnn_cell.RNNCell

class PetridishRNNCell(RNNCell):

    def __init__(
            self,
            num_units,
            layer_info_list,
            state_idx=-1,
            proj_idx=-1,
            num_proj=None,
            hid_to_fs_params=None,
            l_hallu_costs=None,
            initializer=None,
            data_format='channels_first',
            compute_hallu_stats=False,
            h_mask=None,
            x_mask=None):
        """
        """
        self._num_units = num_units
        self.layer_info_list = layer_info_list
        self.state_idx = state_idx
        self.proj_idx = proj_idx
        self.data_format = data_format
        self.compute_hallu_stats = compute_hallu_stats
        self.hallu_record = None
        self.n_calls = 0
        self.hid_to_fs_params = hid_to_fs_params
        self.l_hallu_costs = l_hallu_costs
        self.drop_path_func = None
        self.hallu_record = _init_hallu_record(self.compute_hallu_stats)
        self.h_mask = h_mask
        self.x_mask = x_mask
        self._sum_hallu_cost = 0.

        if num_proj:
            self._num_proj = num_proj
            # one extra dim for hallu costs
            self._output_size = (num_proj, 1)
        else:
            self._num_proj = None
            self._output_size = (num_units, 1)

        if initializer is not None:
            self.initializer = initializer
        else:
            sqrt3 = np.sqrt(0.04)
            self.initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_size

    def get_state_var(self, name, batch_size):
        """
        Create a state for the cell.
        The initial state for the cell must be created by this func.
        The final result of the state should be the same format.
        """
        return tf.get_variable(
            name, [batch_size, self._num_units],
            trainable=False, initializer=tf.constant_initializer(0))

    def get_update_ops(self, state_var, last_state):
        """
        Get update op for setting the state_var to be last_state, which
        is the result of multiple __call__.

        Returns:
        a list of ops for the update ops
        """
        return [state_var.assign(last_state)]

    def get_reset_state_op(self, state_var):
        """
        Returns:
        a list of ops for resetting the state_var that is created by
        self.get_state_var()
        """
        z = tf.zeros_like(state_var)
        return [state_var.assign(z)]

    def split_outputs(self, outputs):
        """
        Args:
        outputs (tuple) : the first term of result of __call__().
        Check the term (O, C) in the returns of __call__
        """
        feats, costs = outputs
        return feats, tf.reduce_sum(costs)

    def _init_layer(self, inputs, state):
        ch_dim = 1
        layer = x_and_h = tf.concat(axis=ch_dim, values=[inputs, state])
        ht = FullyConnected(
            'init_fc',
            layer,
            2 * self._num_units,
            activation=tf.identity,
            use_bias=False)
        h, t = tf.split(ht, 2, axis=ch_dim)
        h = tf.tanh(h)
        t = tf.sigmoid(t)
        init_layer = state + t * (h - state)
        return x_and_h, init_layer

    def __call__(self, inputs, state, scope=None):
        """
        Args:
        inputs (tuple or tensor):
            It's either (X, cost) or X, where X is the inputs tensor and
            cost is some accumulated costs.
            The case of X happens for the first cell in a stack of cells
            The case of (X, cost) happens for the subsequent cells in
            an MultiRNNCell.
        state: should be of the same shape as the result of get_state_var()

        Returns:
        a nested tuple ((O, C), S) where O is B x num_proj, C is B x 1,
        S is B x num_units. O is the outputs used for further inference,
        S is the last state. C is a cost vector.

        Note that (O,C) will be the input of the subsequent cell in a
        multi-rnn cell.
        """
        if isinstance(inputs, tuple):
            # this is for multr nn cell (nn.MultiRNNCell)
            # see tensorflow/python/ops/rnn_cell_impl.py for why
            # the inputs shape is the same as the output shape of this __call__
            inputs, init_cell_hallu_cost = self.split_outputs(inputs)
        else:
            init_cell_hallu_cost = 0.
        batch_size = inputs.get_shape().as_list()[0]
        self.n_calls += 1
        n_layers = len(self.layer_info_list)
        layer_dict = dict()
        l_layers = []
        with tf.variable_scope(scope or type(self).__name__):
            if self.x_mask is not None:
                inputs = inputs * self.x_mask
            if self.h_mask is not None:
                state = state * self.h_mask
            x_and_h, init_layer = self._init_layer(inputs, state)
            cell_hallu_cost = init_cell_hallu_cost
            #init_layer = BatchNorm('bn_init', init_layer)
            for layer_idx in range(n_layers):
                info = self.layer_info_list[layer_idx]
                if layer_idx == 0:
                    layer = x_and_h
                elif layer_idx == 1:
                    layer = init_layer
                else:
                    prev_n_hallu_cost = len(self.l_hallu_costs)
                    layer = construct_layer(
                        name="layer{:03d}".format(info.id),
                        layer_dict=layer_dict,
                        layer_info=info,
                        out_filters=self._num_units,
                        strides=1,
                        data_format=self.data_format,
                        stop_gradient=info.stop_gradient,
                        hid_to_fs_params=self.hid_to_fs_params,
                        drop_path_func=self.drop_path_func,
                        l_hallu_costs=self.l_hallu_costs)
                    # darts suggest using this during search
                    #    bn_after_merge=True)
                    if len(self.l_hallu_costs) > prev_n_hallu_cost:
                        cell_hallu_cost += self.l_hallu_costs[-1]
                layer_dict[info.id] = layer
                l_layers.append(layer)
                self.hallu_record = _update_hallu_record(
                    self.compute_hallu_stats, self.hallu_record,
                    layer_idx, self.layer_info_list, layer_dict)
            # The returned outputs have to be B x 1 at least.
            cell_hallu_cost = (
                tf.ones([batch_size, 1])
                * (cell_hallu_cost / tf.cast(batch_size, tf.float32)))
        return ((l_layers[self.proj_idx], cell_hallu_cost),
                l_layers[self.state_idx])

class PetridishRNNModel(ModelDesc):

    def __init__(self, options):
        super(PetridishRNNModel, self).__init__()
        self.options = options
        self.data_format = options.data_format
        self.net_info = options.net_info
        self.layer_info_list = self.net_info['master']
        self.input_type = _type_str_to_type(options.input_type)
        self.output_type = _type_str_to_type(options.output_type)
        self._input_names = None

        self.bs_per_gpu = self.options.batch_size // self.options.nr_gpu
        self.max_len = options.model_rnn_max_len
        self.has_static_len = options.model_rnn_has_static_len
        self.vocab_size = options.model_rnn_vocab_size
        self.num_units = options.model_rnn_num_units
        self.num_lstms = options.model_rnn_num_lstms
        self.num_proj = options.model_rnn_num_proj
        self.lock_embedding = options.model_rnn_lock_embedding

        ## Five different dropouts. Copy from darts. Originaed from variational dropout
        # locked dropout on input after embed
        self.keep_prob_i = options.model_rnn_keep_prob_i
        # emebdding matrix dropout (ignore certain words)
        self.keep_prob_e = options.model_rnn_keep_prob_e
        # cell hidden state dropout
        self.keep_prob_h = options.model_rnn_keep_prob_h
        # cell input x dropout
        self.keep_prob_x = options.model_rnn_keep_prob_x
        # locked dropout on output
        self.keep_prob = options.model_rnn_keep_prob

        # rnn specific regularization constants
        self.rnn_l2_reg = options.model_rnn_l2_reg
        self.rnn_slowness_reg = options.model_rnn_slowness_reg

        self.init_range = options.model_rnn_init_range
        self.compute_hallu_stats = getattr(options, 'compute_hallu_stats', False)
        self.cells = []
        self.b_dim = 0
        self.t_dim = 1
        self.c_dim = 2

    @property
    def num_classes(self):
        return self.options.num_classes

    @property
    def input_size(self):
        return self.options.input_size

    def optimizer(self):
        return optimizer_common(self.options)

    def cell_mask(self, keep_prob):
        if keep_prob is None or keep_prob >= 1.0:
            return None
        mask = tf.random_uniform(
            shape=[self.bs_per_gpu, self.num_units],
            minval=0, maxval=1, dtype=tf.float32)
        mask = tf.floor(mask + keep_prob) / keep_prob
        return mask

    def _basic_cell(
            self,
            initializer=None,
            hid_to_fs_params=None,
            l_hallu_costs=None):
        is_training = get_current_tower_context().is_training
        if is_training:
            h_mask = self.cell_mask(self.keep_prob_h)
            x_mask = self.cell_mask(self.keep_prob_x)
        else:
            h_mask = x_mask = None
        cell = PetridishRNNCell(
            num_units=self.num_units,
            layer_info_list=self.layer_info_list,
            num_proj=self.num_proj,
            hid_to_fs_params=hid_to_fs_params,
            l_hallu_costs=l_hallu_costs,
            initializer=initializer,
            data_format=self.data_format,
            compute_hallu_stats=self.compute_hallu_stats,
            h_mask=h_mask,
            x_mask=x_mask)
        self.cells.append(cell)
        return cell

    def dropout_embedding_w(self, w, keep_prob):
        """
        Dropout for embedding matrix w.
        The idea is to ignore certain words completely at random
        """
        is_training = get_current_tower_context().is_training
        do_dropout = keep_prob is not None and keep_prob < 1.0 and is_training
        if not do_dropout:
            return w

        # [n_vocab, nhid]
        w_shape = w.get_shape().as_list()
        mask = tf.random_uniform(
            shape=[w_shape[0], 1], minval=0, maxval=1, dtype=tf.float32)
        mask = tf.floor(mask + keep_prob) / keep_prob
        return tf.multiply(mask, w)

    def locked_dropout(self, x, keep_prob):
        """
        Variational (locked) dropout. We make sure
        the drop-out mask is the same at all time steps.
        """
        is_training = get_current_tower_context().is_training
        do_dropout = keep_prob is not None and keep_prob < 1.0 and is_training
        if not do_dropout:
            return x

        x_shape = x.get_shape().as_list()
        x_shape[self.t_dim] = 1
        mask = tf.random_uniform(
            x_shape, minval=0, maxval=1, dtype=tf.float32)
        mask = tf.floor(mask + keep_prob) / keep_prob
        return tf.multiply(mask, x)

    def linear_with_embedding_w(self, x, w):
        """
        Args:
        x: dimension is [-1, embed_size]
        w: dimension is [vocab_size, embed_size]
        """
        b = tf.get_variable(
            'pred/b', [self.vocab_size],
            initializer=tf.constant_initializer(0))
        ret = tf.matmul(x, w, transpose_b=True) + b
        ret = tf.identity(ret, name='linear')
        return ret

    def _embed_input_if_int(self, seq, initializer=None):
        if self.input_type == tf.int32:
            embedding_w = tf.get_variable(
                'embedding/W', [self.vocab_size, self.num_units],
                initializer=initializer)
            masked_w = self.dropout_embedding_w(embedding_w, self.keep_prob_e)
            # B x seqlen x num_units
            seq = tf.nn.embedding_lookup(embedding_w, seq)
        else:
            embedding_w = tf.get_variable(
                'embedding/W', [1, self.num_units],
                initializer=initializer)
            seq = tf.reshape(seq, [-1, 1])
            seq = tf.matmul(seq, embedding_w)
            # B x seqlen x num_units
            seq = tf.reshape(seq, [self.bs_per_gpu, -1, self.num_units])
        return seq, embedding_w

    def _build_hallu_stats_graph(self, cost):
        if not self.compute_hallu_stats:
            return
        l_merged_stats = []
        merged_stats = None
        for celli, cell in enumerate(self.cells):
            cell_hallu_record = getattr(cell, 'hallu_record', None)
            merged_stats = _hallu_stats_graph_merged(
                self.compute_hallu_stats,
                cell_hallu_record,
                cost,
                scope='cell_{}'.format(celli),
                n_calls=cell.n_calls,
                layer_info_list=self.layer_info_list)
            l_merged_stats.append(merged_stats)
        # Merge the results of cells.
        if merged_stats:
            n_stats = len(merged_stats)
            names = get_hallu_stats_output_names(self.layer_info_list, scope='master')
            names = list(map(lambda n: n[:-2], names))
            assert len(names) == n_stats, '{} != {}'.format(len(names), n_stats)
            for i in range(n_stats):
                tf.add_n(
                    [stats[i] for stats in l_merged_stats],
                    name=names[i])

    def inputs(self):
        raise NotImplementedError()

    def build_graph(self, *inputs):
        raise NotImplementedError()

class PetridishRNNSingleOutputModel(PetridishRNNModel):

    def __init__(self, options):
        super(PetridishRNNSingleOutputModel, self).__init__(options)
        self._input_names = ['input', 'input_len', 'label']

    def inputs(self):
        return [
            tf.TensorSpec([None, None], self.input_type, 'input'),
            tf.TensorSpec([None], tf.int32, 'input_len'),
            tf.TensorSpec([None], self.output_type, 'label')
        ]

    def build_graph(self, seq, seq_len, label):
        batch_size = tf.shape(seq)[0]

        with argscope(
                    [
                        Conv2D, Deconv2D, GroupedConv2D, AvgPooling,
                        MaxPooling, BatchNorm, GlobalAvgPooling,
                        ResizeImages, SeparableConv2D
                    ],
                    data_format=self.data_format
                ), \
                argscope(
                    [Conv2D, Deconv2D, GroupedConv2D, SeparableConv2D],
                    activation=tf.identity,
                    use_bias=self.options.use_bias
                ), \
                argscope(
                    [BatchNorm],
                    center=False,
                    scale=False,
                    decay=self.options.batch_norm_decay,
                    epsilon=self.options.batch_norm_epsilon
                ), \
                argscope(
                    [candidate_gated_layer],
                    eps=self.options.candidate_gate_eps
                ):

            initializer = tf.random_uniform_initializer(-self.init_range, self.init_range)
            hid_to_fs_params = _init_feature_select(
                self.layer_info_list, 'master', self.options.feat_sel_lambda)
            seq, embedding_w = self._embed_input_if_int(seq, initializer=initializer)
            basic_cells = [
                self._basic_cell(
                    initializer=initializer,
                    hid_to_fs_params=hid_to_fs_params) for _ in range(self.num_lstms)
            ]
            cells = rnn.MultiRNNCell(basic_cells)
            state = cells.zero_state(batch_size, dtype=tf.float32)

            _, state = tf.nn.dynamic_rnn(
                cells, seq, initial_state=state, sequence_length=seq_len)

            cost = None
            cost, _variables = feature_to_prediction_and_loss(
                'rnn_pred', state, label, self.num_classes,
                self.options.prediction_feature, 1, is_last=True)
            self.cost = tf.identity(cost, name='cost')
            # TODO it is broken rn. because dynamic_rnn is not capatible with
            # hallu stats computation.
            self._build_hallu_stats_graph(self.cost)
            return self.cost

class PetridishRNNInputWiseModel(PetridishRNNModel):

    def __init__(self, options):
        super(PetridishRNNInputWiseModel, self).__init__(options)
        self._input_names = ['input', 'label']
        self.params_to_regularize = '.*/[Wb].*'
        self.state_var_names = [
            'state_{}'.format(k) for k in range(self.num_lstms)]
        self.last_state_var_names = [
            name + '_last' for name in self.state_var_names]
        self.load_ignore_var_names = (
            self.state_var_names + self.last_state_var_names)

    def inputs(self):
        seq_len = self.max_len if self.has_static_len else None
        return [
            tf.TensorSpec([self.bs_per_gpu, self.input_type, seq_len], 'input'),
            tf.TensorSpec([self.bs_per_gpu, seq_len], self.output_type, 'label'),
        ]

    def build_graph(self, seq, tseq):
        batch_size = self.bs_per_gpu
        dynamic_seq_len = tf.shape(seq)[1]
        labels = tf.reshape(tseq, [-1])
        DROPOUT = 0.5
        with argscope(
                    [
                        Conv2D, Deconv2D, GroupedConv2D, AvgPooling,
                        MaxPooling, BatchNorm, GlobalAvgPooling,
                        ResizeImages, SeparableConv2D
                    ],
                    data_format=self.data_format
                ), \
                argscope(
                    [Conv2D, Deconv2D, GroupedConv2D, SeparableConv2D],
                    activation=tf.identity,
                    use_bias=self.options.use_bias
                ), \
                argscope(
                    [BatchNorm],
                    center=False,
                    scale=False,
                    decay=self.options.batch_norm_decay,
                    epsilon=self.options.batch_norm_epsilon
                ), \
                argscope(
                    [candidate_gated_layer],
                    eps=self.options.candidate_gate_eps
                ):

            is_training = get_current_tower_context().is_training
            initializer = tf.random_uniform_initializer(-self.init_range, self.init_range)
            # B x seqlen x hidden
            seq, embedding_w = self._embed_input_if_int(seq, initializer=initializer)
            seq = self.locked_dropout(seq, self.keep_prob_i)
            hid_to_fs_params = _init_feature_select(
                self.layer_info_list, 'master', self.options.feat_sel_lambda)
            l_hallu_costs = []

            self.basic_cells = basic_cells = [
                self._basic_cell(
                   initializer=initializer,
                   hid_to_fs_params=hid_to_fs_params,
                   l_hallu_costs=l_hallu_costs) for _ in range(self.num_lstms)
            ]
            cells = rnn.MultiRNNCell(basic_cells)

            self.state = tuple([
                basic_cells[k].get_state_var(
                    self.state_var_names[k], batch_size) \
                for k in range(self.num_lstms)
            ])
            self.last_state = tuple([
                basic_cells[k].get_state_var(
                    self.last_state_var_names[k] + '_last', batch_size) \
                for k in range(self.num_lstms)
            ])
            self._update_init_state_op = self.update_init_state()
            with tf.control_dependencies([self._update_init_state_op]):
                with tf.variable_scope('RNN', initializer=initializer):
                    outputs, last_state = tf.nn.dynamic_rnn(
                        cells, seq, initial_state=self.state,
                        parallel_iterations=self.max_len
                    )
                    # for the update op
            self._update_last_state_op = self.update_last_state(
                tf.stop_gradient(last_state))
            with tf.control_dependencies([self._update_last_state_op]):
                seqout, sum_hallu_costs = basic_cells[-1].split_outputs(
                    outputs)
                seqout = self.locked_dropout(seqout, self.keep_prob)
                flat_seqout = tf.reshape(seqout, [-1, self.num_units])

            # compute logits and prediction log loss
            if self.lock_embedding:
                logits = self.linear_with_embedding_w(flat_seqout, embedding_w)
            else:
                logits = FullyConnected(
                    'linear',
                    flat_seqout, self.vocab_size,
                    activation=tf.identity)
            logloss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            per_seq_logloss = tf.reduce_sum(
                tf.reshape(logloss, [self.bs_per_gpu, -1]),
                axis=1,
                name="per_seq_sum_logloss")
            cost = tf.truediv(
                tf.reduce_sum(logloss),
                tf.cast(self.bs_per_gpu, tf.float32),
                name='avg_batch_cost')
            float_seq_len = tf.cast(
                dynamic_seq_len, tf.float32, name='seq_len')

            # # tensorpack bullshits. Inferencer must use tensors
            # # so we have to create a tensor ....
            # test_time_udpate = self.update_state(
            #     [per_seq_logloss], name='test_time_update')
            # with tf.control_dependencies([test_time_udpate]):
            #     self._inference_update_tensor = tf.multiply(
            #         cost, 1.0001, name=self._inference_update_tensor_name)

            perpl = tf.identity(
                tf.exp(cost / float_seq_len), name='perplexity')
            add_moving_summary(perpl, cost, float_seq_len)

            # regularization
            if self.rnn_l2_reg:
                cost += (self.rnn_l2_reg * tf.reduce_sum(seqout ** 2) /
                         tf.to_float(self.bs_per_gpu))
            if self.rnn_slowness_reg:
                assert self.t_dim == 1
                all_h_diff = tf.reduce_sum(
                    (seqout[:, 1:, :] - seqout[:, :-1, :]) ** 2)
                cost += (self.rnn_slowness_reg * all_h_diff /
                         tf.to_float(self.bs_per_gpu))
            wd_w = self.options.regularize_const
            if self.params_to_regularize is not None and wd_w:
                wd_cost = wd_w * regularize_cost(
                    self.params_to_regularize, tf.nn.l2_loss)
                wd_cost = tf.identity(wd_cost, name='wd_cost')
                add_moving_summary(wd_cost)
                cost += wd_cost
            cost = tf.identity(cost, name='rnn_reg_cost')
            add_moving_summary(cost)

            # hallucination costs
            if l_hallu_costs:
                sum_hallu_costs = tf.identity(
                    sum_hallu_costs, name='hallu_cost')
                add_moving_summary(sum_hallu_costs)
                cost += sum_hallu_costs
            # this computes some gradient norms
            self._build_hallu_stats_graph(cost)
            # scale the loss according the to sequence length
            self.cost = tf.identity(
                cost * float_seq_len / np.float32(self.max_len), name='cost')
            add_moving_summary(self.cost)
            return self.cost


    def reset_state(self, dependencies=[]):
        """
        Reset the states for starting anew.
        """
        with tf.control_dependencies(dependencies):
            ops = []
            for _cell, _state, _last_state in zip(
                    self.basic_cells, self.state, self.last_state):
                ops.extend(_cell.get_reset_state_op(_state))
                ops.extend(_cell.get_reset_state_op(_last_state))
            return tf.group(*ops, name='reset_state')

    def update_last_state(self, last_state, verbose=False):
        update_state_ops = []
        for k in range(self.num_lstms):
            _cell_updates = self.basic_cells[k].get_update_ops(
                self.last_state[k], last_state[k])
            update_state_ops.extend(_cell_updates)
        if verbose:
            with tf.control_dependencies(update_state_ops):
                vals = [
                    get_global_step_var(),
                    tf.reduce_mean(self.last_state[0]),
                    tf.reduce_mean(last_state[0]),
                ]
                update_state_ops.append(tf.Print(vals[-1], vals))
        return tf.group(*update_state_ops, name='set_last_state')

    def update_init_state(self, verbose=False):
        update_state_ops = []
        for k in range(self.num_lstms):
            _cell_updates = self.basic_cells[k].get_update_ops(
                self.state[k], self.last_state[k])
            update_state_ops.extend(_cell_updates)
        if verbose:
            with tf.control_dependencies(update_state_ops):
                vals = [
                    get_global_step_var(),
                    tf.reduce_mean(self.state[0]),
                    tf.reduce_mean(self.last_state[0]),
                ]
                update_state_ops.append(tf.Print(vals[-1], vals))
        return tf.group(*update_state_ops, name='set_init_state')

    def update_state(self, dependencies=[], verbose=False, name=None):
        """
        Update op for shifting states.
        """
        with tf.control_dependencies(dependencies):
            update_state_ops = []
            for k in range(self.num_lstms):
                _cell_updates = self.basic_cells[k].get_update_ops(
                    self.state[k], self.last_state[k])
                update_state_ops.extend(_cell_updates)
            if verbose:
                c = get_global_step_var()
                update_state_ops.append(tf.Print(c, [c]))
            if name is None:
                name = 'update_state'
            return tf.group(*update_state_ops, name=name)

    def inference_update_tensor(self, name_only):
        if not name_only:
            return self._inference_update_tensor
        return self._inference_update_tensor_name

    _inference_update_tensor_name = 'inf_update_tensor'
