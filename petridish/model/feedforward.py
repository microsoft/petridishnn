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
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger

from anytime_models.models.online_learn import AdaptiveLossWeight
from anytime_models.models.anytime_network import ADALOSS_LS_METHOD

from petridish.info import (
    LayerInfo, LayerInfoList, LayerTypes, CellNetworkInfo
)

from petridish.model.common import (
    DYNAMIC_WEIGHTS_NAME, scope_prediction,
    _data_format_to_ch_dim, feature_to_prediction_and_loss,
    optimizer_common, _type_str_to_type
)
from petridish.model.droppath import DropPath
from petridish.model.hallu import (
    _init_hallu_record, _update_hallu_record, _hallu_stats_graph,
    _hallu_stats_graph_merged, get_hallu_stats_output_names
)
from petridish.model.layer import (
    construct_layer, candidate_gated_layer, _init_feature_select,
    initial_convolution, _reduce_prev_layer
)

class PetridishBaseCell(object):

    def __init__(
            self,
            layer_info_list,
            data_format='channels_first',
            compute_hallu_stats=False,
            drop_path_func=None,
            hid_to_fs_params=None,
            l_hallu_costs=None):

        self.layer_info_list = layer_info_list
        self.data_format = data_format
        self.ch_dim = _data_format_to_ch_dim(data_format)
        self.compute_hallu_stats = compute_hallu_stats
        self.hallu_record = _init_hallu_record(self.compute_hallu_stats)
        self.n_calls = 0
        self.auto_cat_unused = False

        #droppath specific
        self.drop_path_func = drop_path_func
        # feature select param
        self.hid_to_fs_params = hid_to_fs_params
        # hallucination statistics
        self.l_hallu_costs = l_hallu_costs

    def __call__(self, inputs, out_filters, stride=1, non_input_layer_idx=-1, scope=None):
        """
        Args:
        inputs (list of Tensor) : a list of input layers to the cell. The last
            input is assumed to be the most recent input.

        """
        n_inputs = len(inputs)
        n_layers = len(self.layer_info_list)
        layer_dict = dict()
        used_layers = set()
        input_layers = set()
        with tf.variable_scope(scope):
            # input preprocessing
            for layer_idx in range(n_inputs):
                info = self.layer_info_list[layer_idx]
                assert LayerInfo.is_input(info), info
                used_layers.add(info.id)
                input_layers.add(info.id)
                layer_dict[info.id] = inputs[layer_idx]

            # construction of layers
            for layer_idx in range(n_inputs, n_layers):
                info = self.layer_info_list[layer_idx]
                # compute stride of each op
                l_strides = stride
                if stride > 1:
                    l_strides = [1] * len(info.inputs)
                    for idx, in_id in enumerate(info.inputs):
                        if in_id in input_layers:
                            l_strides[idx] = stride

                layer = construct_layer(
                    "layer{:03d}".format(info.id),
                    layer_dict,
                    info,
                    out_filters,
                    l_strides,
                    self.data_format,
                    info.stop_gradient,
                    drop_path_func=self.drop_path_func,
                    non_input_layer_idx=non_input_layer_idx,
                    hid_to_fs_params=self.hid_to_fs_params,
                    l_hallu_costs=self.l_hallu_costs)

                layer_dict[info.id] = layer
                # update info for upcoming constructions.
                for in_id in info.inputs:
                    used_layers.add(in_id)

                # hallucination record update
                self.hallu_record = _update_hallu_record(
                    self.compute_hallu_stats, self.hallu_record, layer_idx,
                    self.layer_info_list, layer_dict)
            #end for each layer_idx

            # cat all unused -- this is done at the layer_info stage. Remove later.
            l_unused = [layer_dict[info.id] for info in self.layer_info_list \
                if not info.id in used_layers]
            assert len(l_unused) <= 1, \
                ('Programming bug. Unused should be handled at layer_info level. '
                 'The unused are {}'.format(l_unused))
            #if len(l_unused) > 1:
            #    layer = tf.concat(l_unused, axis=self.ch_dim, name='cat_unused')
        self.n_calls += 1
        return layer

class PetridishModel(ModelDesc):
    """
       Base Petridish model
    """
    def __init__(self, options):
        super(PetridishModel, self).__init__()
        self.options = options

        # Classification info
        self.prediction_feature = options.prediction_feature
        self.out_filters = options.init_channel
        self.stem_channel_rate = options.stem_channel_rate
        self.data_format = options.data_format

        # LayerInfoList as a record of the mathematical graph
        self.net_info = options.net_info
        self.master = self.net_info.master
        self.is_cell_based = self.net_info.is_cell_based()
        self.n_layers = len(self.master)
        self.n_aux_preds = sum([int(x.aux_weight > 0) for x in self.master])

        self.ch_dim = _data_format_to_ch_dim(self.data_format)
        self.params_to_regularize = None

        self.compute_hallu_stats = False
        if hasattr(options, 'compute_hallu_stats'):
            self.compute_hallu_stats = options.compute_hallu_stats

    @property
    def num_classes(self):
        return self.options.num_classes

    @property
    def input_size(self):
        return self.options.input_size


    def inputs(self):
        raise NotImplementedError("A derived class should implement me...ow")

    def _preprocess_data(self, inputs):
        return inputs[0], inputs[1]

    def build_graph(self, *inputs):
        # Dynamic weighting for multiple predictions
        if self.options.ls_method == ADALOSS_LS_METHOD:
            dynamic_weights = tf.get_variable(
                DYNAMIC_WEIGHTS_NAME, (self.n_aux_preds,),
                tf.float32, trainable=False,
                initializer=tf.constant_initializer([1.0]*self.n_aux_preds))
            for i in range(self.n_aux_preds):
                weight_i = tf.identity(
                    dynamic_weights[i], 'weight_{:02d}'.format(i))
                add_moving_summary(weight_i)

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
                    momentum=float(self.options.batch_norm_decay),
                    epsilon=float(self.options.batch_norm_epsilon)
                ), \
                argscope(
                    [candidate_gated_layer],
                    eps=self.options.candidate_gate_eps
                ):

            # regularization initialization
            if self.options.regularize_coef == 'const':
                wd_w = self.options.regularize_const
            elif self.options.regularize_coef == 'decay':
                wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                                  480000, 0.2, True)


            # Network-level objects / information
            n_inputs = self.master.num_inputs()
            drop_path_func = DropPath(
                drop_path_keep_prob=self.options.drop_path_keep_prob,
                max_train_steps=self.options.max_train_steps,
                total_depth=self.n_layers - n_inputs)

            l_hallu_costs = []
            # cell dictionary
            self.op_to_cell = dict()
            for cname in self.net_info.cell_names:
                hid_to_fs_params = _init_feature_select(
                    self.net_info[cname], cname, self.options.feat_sel_lambda)
                if cname == 'master':
                    # since master has additional duties like down_sampling,
                    # aux prediction, accumulating hallu stats, etc,
                    # master is not built from cell
                    master_hid_to_fs_params = hid_to_fs_params
                    hallu_record = _init_hallu_record(self.compute_hallu_stats)
                    continue

                self.op_to_cell[cname] = PetridishBaseCell(
                    self.net_info[cname],
                    self.data_format,
                    self.compute_hallu_stats,
                    drop_path_func=drop_path_func,
                    hid_to_fs_params=hid_to_fs_params,
                    l_hallu_costs=l_hallu_costs)

            l_layers = [None] * self.n_layers
            layer_dict = dict()
            out_filters = self.out_filters

            # on-GPU(device) preprocessing for mean/var, casting, embedding, init conv
            layer, label = self._preprocess_data(inputs)

            for layer_idx in range(n_inputs):
                info = self.master[layer_idx]
                layer_dict[info.id] = layer #if layer_idx + 1 == n_inputs else None

            for layer_idx in range(n_inputs, self.n_layers):
                info = self.master[layer_idx]
                layer_id_str = "layer{:03d}".format(info.id)
                strides = 1
                if info.down_sampling:
                    out_filters *= 2
                    strides = 2
                # preprocess all inputs to match the most recent layer
                # in h/w and out_filters in ch_dim
                #if not self.is_cell_based:
                with tf.variable_scope('pre_'+layer_id_str):
                    orig_dict = dict()
                    for input_id in info.inputs:
                        in_l = layer_dict[input_id]
                        orig_dict[input_id] = in_l
                        layer_dict[input_id] = _reduce_prev_layer(
                            in_l, input_id, layer, out_filters,
                            self.data_format, hw_only=False)
                layer = construct_layer(
                    layer_id_str, layer_dict, info, out_filters, strides,
                    self.data_format, info.stop_gradient,
                    op_to_cell=self.op_to_cell,
                    drop_path_func=drop_path_func,
                    non_input_layer_idx=layer_idx - n_inputs,
                    hid_to_fs_params=master_hid_to_fs_params,
                    l_hallu_costs=l_hallu_costs
                )

                # store info for future compute
                layer_dict[info.id] = layer
                l_layers[layer_idx] = layer
                hallu_record = _update_hallu_record(
                    self.compute_hallu_stats,
                    hallu_record, layer_idx, self.master, layer_dict)
                #if not self.is_cell_based and self.options.use_local_reduction:
                if self.options.use_local_reduction:
                    # reset the reduction layers in dict. So each layer
                    # uses its own reduction
                    for input_id in orig_dict:
                        layer_dict[input_id] = orig_dict[input_id]
            # end for layer wise feature construction.

            # build aux predictions
            total_cost = 0.0
            wd_cost = 0.0
            anytime_idx = -1
            for layer_idx, layer in enumerate(l_layers):
                # aux prediction
                info = self.master[layer_idx]
                cost_weight = info.aux_weight

                if cost_weight > 0:
                    anytime_idx += 1

                    scope_name = scope_prediction(info.id)
                    cost, variables = feature_to_prediction_and_loss(
                        scope_name, layer, label,
                        self.num_classes, self.prediction_feature,
                        ch_dim=self.ch_dim,
                        label_smoothing=self.options.label_smoothing,
                        dense_dropout_keep_prob=self.options.dense_dropout_keep_prob,
                        is_last=(layer_idx + 1 == len(l_layers)))

                    # record the cost for the use of online learners.
                    cost_i = tf.identity(cost, name='anytime_cost_{:02d}'.format(anytime_idx))

                    # decide whether to use static or dynmic weights
                    if self.options.ls_method == ADALOSS_LS_METHOD:
                        cost_weight = dynamic_weights[anytime_idx]
                    total_cost += cost_weight * cost_i

                    # regularize variable in linear predictors
                    # (have to do this separately here because
                    # we need unregularized losses for cost_weights)
                    for var in variables:
                        wd_cost += cost_weight * wd_w * tf.nn.l2_loss(var)
                # end if aux_weight > 0
            # end for each layer

            # regularization, cost
            if self.params_to_regularize is not None:
                wd_cost += wd_w * regularize_cost(self.params_to_regularize, tf.nn.l2_loss)
            wd_cost = tf.identity(wd_cost, name='wd_cost')
            total_cost = tf.identity(total_cost, name='sum_losses')
            add_moving_summary(total_cost, wd_cost)
            if l_hallu_costs:
                hallu_total_cost = tf.add_n(l_hallu_costs, name='hallu_total_cost')
                add_moving_summary(hallu_total_cost)
                self.cost = tf.add_n([total_cost, wd_cost, hallu_total_cost], name='cost')
            else:
                self.cost = tf.add_n([total_cost, wd_cost], name='cost')

            # hallu stats
            for cname in self.net_info.cell_names:
                if cname == 'master':
                    _hallu_stats_graph(
                        self.compute_hallu_stats, hallu_record, self.cost, scope=cname)
                    continue
                cell = self.op_to_cell.get(cname, None)
                cell_hallu_record = getattr(cell, 'hallu_record', None)
                _hallu_stats_graph_merged(
                    self.compute_hallu_stats, cell_hallu_record,
                    self.cost, scope=cname, n_calls=cell.n_calls,
                    layer_info_list=cell.layer_info_list)
            return self.cost


    def optimizer(self):
        return optimizer_common(self.options)

    def compute_loss_select_callbacks(self):
        """
            A list of callbacks for the trainer for updating dyanmic weights
        """
        if self.options.ls_method == ADALOSS_LS_METHOD:
            loss_names = ['tower0/anytime_cost_{:02d}'.format(i) \
                    for i in range(self.n_aux_preds)]
            dynamic_weights_name = '{}:0'.format(DYNAMIC_WEIGHTS_NAME)
            online_learn_cb = AdaptiveLossWeight(
                self.n_aux_preds,
                dynamic_weights_name, loss_names,
                gamma=self.options.adaloss_gamma,
                update_per=self.options.adaloss_update_per,
                momentum=self.options.adaloss_momentum,
                final_extra=self.options.adaloss_final_extra)
            online_learn_cbs = [online_learn_cb]
        else:
            online_learn_cbs = []
        return online_learn_cbs

class RecognitionModel(PetridishModel):

    def __init__(self, options):
        super(RecognitionModel, self).__init__(options)
        # input types (int8 vs float32)
        self.input_type = _type_str_to_type(options.input_type)
        if self.options.do_mean_std_gpu_process:
            if not hasattr(self.options, 'mean'):
                raise Exception('gpu_graph expects mean but it is not in the options')
            if not hasattr(self.options, 'std'):
                raise Exception('gpu_graph expects std, but it is not in the options')

        # data format; input dimensions
        self.ch_dim = _data_format_to_ch_dim(self.data_format)
        self.params_to_regularize = '.*conv.*/W.*'
        self._input_names = ['input', 'label']

    def inputs(self):
        return [
            tf.TensorSpec(
                [None, self.input_size, self.input_size, 3], self.input_type,
                self._input_names[0]),
            tf.TensorSpec([None], tf.int32, self._input_names[1])
        ]

    def _preprocess_data(self, inputs):
        image = inputs[0]
        label = inputs[1]
        # Common GPU side preprocess (uint8 -> float32), mean std, NCHW transpose.
        if self.input_type == tf.uint8:
            image = tf.cast(image, tf.float32) * (1.0 / 255)
        if self.options.do_mean_std_gpu_process:
            if self.options.mean is not None:
                image = image - tf.constant(self.options.mean, dtype=tf.float32)
            if self.options.std is not None:
                image = image / tf.constant(self.options.std, dtype=tf.float32)
        if self.data_format == 'channels_first':
            image = tf.transpose(image, [0, 3, 1, 2])
        self.dynamic_batch_size = tf.identity(tf.shape(image)[0], name='dynamic_batch_size')
        tf.reduce_mean(image, name='dummy_image_mean')

        layer = initial_convolution(
            image, int(self.out_filters * self.options.stem_channel_rate),
            self.options.s_type, name="stem_conv")
        return layer, label

class MLPModel(PetridishModel):

    def __init__(self, options):
        super(MLPModel, self).__init__(options)
        self.input_types = options.mlp_input_types
        self.input_dims = options.mlp_input_dims
        self.feat_means = options.mlp_feat_means
        self.feat_stds = options.mlp_feat_stds
        self._input_names = ['input_{}'.format(i) for i in range(len(self.input_types) - 1)]
        self._input_names.append('label')

    def inputs(self):
        ret = []
        for t, d, n in zip(self.input_types, self.input_dims, self._input_names):
            if d is None or t == tf.int32:
                shape = [None]
            else:
                shape = [None, d]
            ret.append(tf.TensorSpec(shape, t, n))
        return ret

    def _preprocess_data(self, inputs):
        self.dynamic_batch_size = tf.identity(
            tf.shape(inputs[0])[0], name='dynamic_batch_size')
        feats = inputs[0:len(inputs)-1]
        label = inputs[-1]

        def _target_dim(dim):
            if dim <= 2:
                return 1
            return max(1, min(dim, int(np.log2(dim) + 1)))

        new_feats = []
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        for fi, feat, dim, t, m, std in zip(
                range(len(feats)),
                feats, self.input_dims, self.input_types,
                self.feat_means, self.feat_stds):
            if t == tf.int32:
                if dim == 1:
                    feat = tf.cast(feat, tf.float32) - 0.5
                else:
                    with tf.variable_scope('init_embed_{}'.format(fi)):
                        embedding_w = tf.get_variable(
                            'embedding_w',
                            [dim, _target_dim(dim)], initializer=initializer)
                        feat = tf.nn.embedding_lookup(
                            embedding_w, tf.reshape(feat, [-1])) # B x hiddensize
                new_feats.append(feat)
            else:
                if len(feat.get_shape().as_list()) == 1:
                    feat = tf.reshape(feat, [-1, 1])
                new_feats.append((feat - m) / std)
            #logger.info(feat.get_shape().as_list())
        cat_feat = tf.concat(axis=1, values=new_feats)
        out_filters = int(self.out_filters * self.options.stem_channel_rate)
        return FullyConnected('init_fc', cat_feat, out_filters), label
