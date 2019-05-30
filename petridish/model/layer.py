import numpy as np
import tensorflow as tf
import re

from tensorpack.models import (
    BatchNorm, BNReLU, Conv2D, AvgPooling, MaxPooling,
    LinearWrap, GlobalAvgPooling, FullyConnected, regularize_cost,
    Deconv2D, GroupedConv2D, ResizeImages, SeparableConv2D, Dropout)
from tensorpack.models.common import layer_register
from tensorpack.utils import logger
from tensorpack.utils.argtools import get_data_format

from petridish.info import (
    LayerInfo, LayerInfoList, LayerTypes, CellNetworkInfo
)
from petridish.model.common import (
    _get_dim, _data_format_to_ch_dim
)

def construct_layer(
        name,
        layer_dict,
        layer_info,
        out_filters,
        strides,
        data_format,
        stop_gradient=None,
        op_to_cell=None,
        drop_path_func=None,
        non_input_layer_idx=None,
        hid_to_fs_params=None,
        l_hallu_costs=None,
        bn_after_merge=False):
    """
    Args:
        name (str) : name of this layer to construct
        layer_dict (dict) : a map from layer id to layer tenors.
        layer_info (LayerInfo): the layer that we are to construct
        out_filters : output number of filters.
        strides : whether to take a strides in the operations
        data_format : 'channel_first' or 'channel_last'
        stop_gradient : whether to stop gradient on inputs
        op_to_cell : a dict from op name (cell name) to the cell object.
        drop_path_func : a function object for computing drop path
        non_input_layer_idx :
            index for computing drop path with for cell based search
        hid_to_fs_params :
            a map from hallu id to tuples of params for feature selection
        l_hallu_costs :
            a list to update, in order to contain the costs incurred by hallu.
        bn_after_merge (bool) :
            whether to batch normalize after the merge op.
            Usually cnn uses False, and rnn uses True.

    Side Effects on inputs:
        1. Update l_hallu_costs, if the constructed layer triggers hallu costs.
        2. Other inputs are unaffected.

    Return:
        a tensor that represents layer
    """
    if op_to_cell is not None:
        # cell network has merge_op as the cell name (str)
        cell = op_to_cell.get(layer_info.merge_op, None)
        if cell:
            inputs = [layer_dict[in_id] for in_id in layer_info.inputs]
            layer = cell(inputs, out_filters, strides, non_input_layer_idx, name)
            return layer

    with tf.variable_scope(name):
        ch_dim = _data_format_to_ch_dim(data_format)
        n_inputs = len(layer_info.inputs)
        new_layer = []
        # stride may be a val for each input.
        if isinstance(strides, list):
            if len(strides) != n_inputs:
                raise ValueError("Confusing strides at info {}".format(layer_info))
            l_strides = strides
        else:
            l_strides = [strides] * n_inputs
        # stop gradient may be a val for each input
        if not isinstance(stop_gradient, list):
            l_to_stop = [stop_gradient] * n_inputs
        # operation names for children
        ops_ids = None
        if layer_info.extra_dict is not None:
            ops_ids = layer_info.extra_dict.get('ops_ids', None)
        ops_ids = ops_ids if ops_ids else list(range(n_inputs))

        for input_i, layer_id, operation, strides, to_stop in zip(
                ops_ids, layer_info.inputs, layer_info.input_ops,
                l_strides, l_to_stop):
            layer = layer_dict[layer_id]
            scope = "op_{}".format(input_i)
            with tf.variable_scope(scope):
                if to_stop == LayerTypes.STOP_GRADIENT_HARD:
                    layer = tf.stop_gradient(layer)
                elif to_stop == LayerTypes.STOP_GRADIENT_SOFT:
                    layer = finalized_gated_layer('soft_stop', layer)
                elif to_stop != LayerTypes.STOP_GRADIENT_NONE:
                    raise ValueError("Unknown stop_gradient value from info {}".format(
                        layer_info))
                layer = apply_operation(
                    layer, operation, out_filters, strides, data_format)
                if LayerTypes.do_drop_path(operation) and drop_path_func:
                    layer = drop_path_func(layer, non_input_layer_idx)
                new_layer.append(layer)

        if len(new_layer) == 1:
            layer = new_layer[0]
        elif len(new_layer) > 1:
            LT = LayerTypes
            merge_op = layer_info.merge_op
            if merge_op in [LT.MERGE_WITH_CAT_PROJ, LT.MERGE_WITH_CAT]:
                layer = tf.concat(new_layer, axis=ch_dim, name='cat_feats')
                if bn_after_merge:
                    layer = BatchNorm('bn_after_merge', layer)
                if merge_op == LT.MERGE_WITH_CAT_PROJ:
                    layer = projection_layer('post_cat', layer, out_filters, ch_dim)
            else:
                # The merge op is not concat-like, so we make sure all inputs are of the
                # same channel first.
                ch_ref = 0
                for layer in new_layer:
                    ch_ref = max(_get_dim(layer, ch_dim), ch_ref)

                # The following block project the new layers
                # to be of the same channel (ch_ref),
                # so they can be add/mul together.
                # However, if the new layer is a gated layer,
                # we need to project the input of
                # the gated layer instead, because we need the shape
                # of the sum tensor and its gated inputs to have the
                # same shape for gradient/tensor comparison. The gate
                # needs to be right before the sum for the gradient/tensor
                # to be non-trivial.
                for li, layer in enumerate(new_layer):
                    if _get_dim(layer, ch_dim) != ch_ref:
                        if hasattr(layer, 'pre_gate_layer'):
                            pre_gate_layer = layer.pre_gate_layer
                            layer = projection_layer(
                                'pre_sum_{}'.format(li),
                                pre_gate_layer, ch_ref, ch_dim)
                            with tf.variable_scope('pre_sum_gate_{}'.format(li)):
                                new_layer[li] = apply_operation(
                                    layer, layer_info.input_ops[li],
                                    ch_ref, 1, data_format)
                                # since the apply_operation is definitely no_param,
                                # it does not require drop path.
                        else:
                            new_layer[li] = projection_layer(
                                'pre_sum_{}'.format(li),
                                layer, ch_ref, ch_dim)

                if merge_op in [LT.MERGE_WITH_SUM, LT.MERGE_WITH_AVG]:
                    layer = tf.add_n(new_layer, name='sum_feats')
                    if merge_op == LT.MERGE_WITH_AVG:
                        layer = tf.div(
                            layer, np.float32(len(new_layer)), name='avg_feats')

                elif merge_op == LT.MERGE_WITH_WEIGHTED_SUM:
                    if layer_info.is_candidate:
                        omega, l1_lambda = hid_to_fs_params[layer_info.is_candidate]
                        layer = feature_selection_layer(
                            'feature_select', new_layer, omega, l1_lambda, l_hallu_costs)
                    else:
                        if layer_info.extra_dict is None:
                            fs_omega = None
                        else:
                            fs_omega = layer_info.extra_dict.get('fs_omega', None)
                        layer = weighted_sum_layer(
                            'weighted_sum', new_layer, fs_omega=fs_omega)

                elif merge_op == LT.MERGE_WITH_MUL:
                    layer = new_layer[0]
                    for idx, layer2 in enumerate(new_layer[1:]):
                        layer = tf.multiply(layer, layer2, name='mul_{}'.format(idx))

                elif merge_op == LT.MERGE_WITH_SOFTMAX:
                    logits = []
                    for li, layer in enumerate(new_layer):
                        with tf.variable_scope('path_choice_{}'.format(li)):
                            n_dim = len(layer.get_shape().as_list())
                            if n_dim > 2:
                                layer = GlobalAvgPooling('softmax_gap', layer) #batch x ch_ref
                            logit = FullyConnected(
                                'softmax_linear', layer, 1, activation=tf.identity)
                            logit = tf.reshape(logit, [-1]) # batch
                            logits.append(logit)
                    logits = tf.stack(logits, axis=1) # batch x len(new_layer)
                    probs = tf.nn.softmax(logits, axis=1) # batch
                    for li, layer in enumerate(new_layer):
                        new_layer[li] = probs[:, li] * layer
                    layer = tf.add_n(new_layer, name='sum_feats')

                else:
                    raise ValueError("Unknown merge operation in info {}".format(
                        layer_info))

                # batch normalization for all non-concat-based merges.
                if bn_after_merge:
                    layer = BatchNorm('bn_after_merge', layer)
            # end else for concat vs non-cat merges.
        else:
            raise ValueError("Layer {} has empty input edges. The info: {}".format(
                name, layer_info))
        return layer

def apply_operation(layer, operation, out_filters, strides, data_format):
    """
    Apply primitive operation to a layer tensor.

    Args:
        layer (tensor) : tensor of the layer to apply the op to.
        operation (int from LayerTypes) : operation copied
        out_filters : number of output filters
        stride : Strides for the operation if applicable
        data_format : data format of the tensor input
    Returns:
        a Tensor that is the result of the operation.
    """
    ch_dim = _data_format_to_ch_dim(data_format)
    if operation == LayerTypes.NOT_EXIST:
        return None

    elif operation == LayerTypes.IDENTITY:
        if strides == 1:
            return tf.identity(layer, name='id')
        else:
            return _factorized_reduction('id_reduction', layer, out_filters, data_format)

    elif operation == LayerTypes.RESIDUAL_LAYER:
        return residual_layer('res', layer, out_filters, strides, data_format)

    elif operation == LayerTypes.RESIDUAL_BOTTLENECK_LAYER:
        return residual_bottleneck_layer('res_btl', layer, out_filters, strides, data_format)

    elif operation == LayerTypes.CONV_1:
        layer = tf.nn.relu(layer)
        layer = Conv2D('conv1x1', layer, out_filters, 1, strides=strides)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.CONV_3:
        layer = tf.nn.relu(layer)
        layer = Conv2D('conv3x3', layer, out_filters, 3, strides=strides)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.SEPARABLE_CONV_3:
        layer = tf.nn.relu(layer)
        layer = Conv2D('conv1x1', layer, out_filters, 1, strides=1, activation=BNReLU)
        layer = SeparableConv2D('sep_conv3x3_1', layer, out_filters, 3, strides=strides)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.SEPARABLE_CONV_5:
        layer = tf.nn.relu(layer)
        layer = Conv2D('conv1x1', layer, out_filters, 1, strides=1, activation=BNReLU)
        layer = SeparableConv2D('sep_conv5x5_1', layer, out_filters, 5, strides=strides)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.SEPARABLE_CONV_3_2:
        layer = tf.nn.relu(layer)
        layer = SeparableConv2D(
            'sep_conv3x3_1', layer, out_filters, 3, strides=strides, activation=BNReLU)
        layer = SeparableConv2D('sep_conv3x3_2', layer, out_filters, 3, strides=1)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.SEPARABLE_CONV_5_2:
        layer = tf.nn.relu(layer)
        layer = SeparableConv2D(
            'sep_conv5x5_1', layer, out_filters, 5, strides=strides, activation=BNReLU)
        layer = SeparableConv2D('sep_conv5x5_2', layer, out_filters, 5, strides=1)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.SEPARABLE_CONV_7_2:
        layer = tf.nn.relu(layer)
        layer = SeparableConv2D(
            'sep_conv7x7_1', layer, out_filters, 7, strides=strides, activation=BNReLU)
        layer = SeparableConv2D('sep_conv7x7_2', layer, out_filters, 7, strides=1)
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.DILATED_CONV_3:
        if strides > 1:
            layer = tf.nn.relu(layer)
            layer = _factorized_reduction(
                'dil_reduction', layer, out_filters, data_format)
        layer = tf.nn.relu(layer)
        layer = SeparableConv2D(
            'dil_conv3x3', layer, out_filters, 3,
            strides=1, dilation_rate=(2, 2))
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.DILATED_CONV_5:
        if strides > 1:
            layer = tf.nn.relu(layer)
            layer = _factorized_reduction(
                'dil_reduction', layer, out_filters, data_format)
        layer = tf.nn.relu(layer)
        layer = SeparableConv2D(
            'dil_conv5x5', layer, out_filters, 5,
            strides=1, dilation_rate=(2, 2))
        layer = BatchNorm('bn', layer)
        return layer

    elif operation == LayerTypes.MAXPOOL_3x3:
        layer = MaxPooling('maxpool', layer, pool_size=3, strides=strides, padding='same')
        ch_in = _get_dim(layer, ch_dim)
        if ch_in != out_filters:
            projection_layer('proj_maxpool', layer, out_filters, ch_dim)
        return layer

    elif operation == LayerTypes.AVGPOOL_3x3:
        layer = AvgPooling('avgpool', layer, pool_size=3, strides=strides, padding='same')
        ch_in = _get_dim(layer, ch_dim)
        if ch_in != out_filters:
            projection_layer('proj_avgpool', layer, out_filters, ch_dim)
        return layer

    elif operation in [
            LayerTypes.GATED_LAYER, LayerTypes.ANTI_GATED_LAYER,
            LayerTypes.NO_FORWARD_LAYER]:
        if strides > 1:
            layer = _factorized_reduction(
                'gate_reduction', layer, out_filters, data_format)
        # this is important for computing hallucination statistics.
        pre_gate_layer = layer
        if operation == LayerTypes.GATED_LAYER:
            layer = finalized_gated_layer('gated_layer', layer)
        elif operation == LayerTypes.NO_FORWARD_LAYER:
            layer = candidate_gated_layer('gated_layer', layer)
        else:
            layer = finalized_gated_layer('anti_gated_layer', layer, init_val=1.0)
        layer.pre_gate_layer = pre_gate_layer
        return layer

    elif operation == LayerTypes.FullyConnected:
        layer = FullyConnected('fully_connect', layer, out_filters)
        layer = tf.nn.relu(layer, 'relu')
        return layer

    elif operation in [
            LayerTypes.FC_TANH_MUL_GATE, LayerTypes.FC_SGMD_MUL_GATE,
            LayerTypes.FC_RELU_MUL_GATE, LayerTypes.FC_IDEN_MUL_GATE]:
        ht = FullyConnected(
            'fully_connect', layer, 2 * out_filters,
            activation=tf.identity, use_bias=False)
        ch_dim = 1
        h, t = tf.split(ht, 2, axis=ch_dim)
        t = tf.sigmoid(t)
        if operation == LayerTypes.FC_TANH_MUL_GATE:
            h = tf.tanh(h)
        elif operation == LayerTypes.FC_SGMD_MUL_GATE:
            h = tf.sigmoid(h)
        elif operation == LayerTypes.FC_RELU_MUL_GATE:
            h = tf.nn.relu(h)
        elif operation == LayerTypes.FC_IDEN_MUL_GATE:
            h = tf.identity(h)
        # add residual
        if _get_dim(layer, ch_dim) != out_filters:
            layer = FullyConnected(
                'proj_prev', layer, out_filters, activation=tf.identity)
        layer = layer + t * (h - layer)
        return layer

    elif operation == LayerTypes.MLP_RESIDUAL_LAYER:
        raise NotImplementedError("MLP residual layer is not yet implemented")

    else:
        raise NotImplementedError("Have not implemented operation {}".format(operation))

def projection_layer(name, layer, out_filters, ch_dim, id_mask_slice=None):
    with tf.variable_scope(name):
        n_dim = len(layer.get_shape().as_list())
        if n_dim == 4:
            layer = tf.nn.relu(layer)
            layer = Conv2D('conv1x1_proj', layer, out_filters, 1, strides=1, activation=tf.identity)
            layer = BatchNorm('bn_proj', layer)
        elif n_dim == 2:
            layer = tf.nn.relu(layer)
            layer = FullyConnected('fc_proj', layer, out_filters, activation=tf.identity)
        else:
            raise ValueError("Projection cannot handle tensor of dim {}".format(n_dim))
        return layer

@layer_register(log_shape=False)
def candidate_gated_layer(layer, eps=None):
    if eps is None:
        eps = 0.0
    layer = layer * eps
    layer_sg = tf.stop_gradient(layer)
    gated_layer = tf.identity(- layer_sg + layer, name='result')
    return gated_layer

def finalized_gated_layer(name, layer, init_val=0.0):
    with tf.variable_scope(name):
        fw = tf.get_variable(
            'f_w', [], dtype=tf.float32,
            initializer=tf.constant_initializer(init_val))
        gated_layer = fw * layer
        return gated_layer

def _init_feature_select(layer_info_list, cell_name, feat_sel_lambda=None):
    hid_to_params = dict()
    with tf.variable_scope(cell_name + '_feature_select'):
        for info in layer_info_list:
            hid = info.is_candidate
            if hid > 0 and info.merge_op == LayerTypes.MERGE_WITH_WEIGHTED_SUM:
                n_inputs = len(info.inputs)
                #for backward compatibility:
                if feat_sel_lambda is None:
                    feat_sel_lambda = 1e-2 / n_inputs
                with tf.variable_scope('hallu_{}'.format(hid)):
                    omega = tf.get_variable(
                        'omega_init', [n_inputs], dtype=tf.float32,
                        initializer=tf.constant_initializer(1. / n_inputs))
                    l1_lambda = tf.get_variable(
                        'l1_lambda', [], dtype=tf.float32,
                        initializer=tf.constant_initializer(feat_sel_lambda),
                        trainable=False)
                    hid_to_params[hid] = (omega, l1_lambda)
    return hid_to_params

def feature_selection_layer(name, layers, omega, l1_lambda, l_hallu_costs):
    """
    Args:
        name :
        layers :
        omega : vector variable for lienar combining operations.
        l1_lambda : scalar variable (that is not learnable)
        l_hallu_costs : a list to modify to contain the costs incurred by hallu
    """
    with tf.variable_scope(name):
        to_sum = [layer * omega[idx] for idx, layer in enumerate(layers)]
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'pre_gate_layer'):
                to_sum[idx].pre_gate_layer = layer.pre_gate_layer
        layer = tf.add_n(to_sum, name='weighted_sum_feats')

        # L2 norm regularization causes bad convergence... will investigate later.
        #n_dim = len(layer.get_shape().as_list())
        #l2_norm = 0.5 * tf.reduce_sum(layer **2, axis=list(range(1,n_dim)))
        #mean_norm = tf.reduce_mean(l2_norm)
        #l_hallu_costs.append(mean_norm)
        lasso_loss = l1_lambda * tf.reduce_sum(tf.abs(omega))
        l_hallu_costs.append(lasso_loss)
        return layer

def weighted_sum_layer(name, layers, fs_omega):
    with tf.variable_scope(name):
        n_layers = len(layers)
        fs_omega = fs_omega if fs_omega is not None else 1. / n_layers
        omega = tf.get_variable(
            'omega', [n_layers], dtype=tf.float32,
            initializer=tf.constant_initializer(fs_omega))
        to_sum = [layer * omega[idx] for idx, layer in enumerate(layers)]
        layer = tf.add_n(to_sum, name='weighted_sum_feats')
        return layer

def residual_layer(name, l, out_filters, strides, data_format):
    ch_out = out_filters
    data_format = get_data_format(data_format, keras_mode=False)
    ch_dim = 3 if data_format == 'NHWC' else 1
    ch_in = _get_dim(l, ch_dim)

    l_in = l
    with tf.variable_scope('{}.0'.format(name)):
        l = BNReLU(l)
        l = SeparableConv2D('conv1', l, ch_out, 3, strides=strides, activation=BNReLU)
        l = SeparableConv2D('conv2', l, ch_out, 3)
        # The second conv need to be BN before addition.
        l = BatchNorm('bn2', l)

        shortcut = l_in
        if strides > 1:
            shortcut = AvgPooling('pool', shortcut, 2)
        if ch_in < ch_out:
            pad_paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]
            pad_width = (ch_out - ch_in)
            pad_paddings[ch_dim] = [0, pad_width]
            shortcut = tf.pad(shortcut, pad_paddings)
        elif ch_in > ch_out:
            if data_format == 'NHWC':
                shortcut1 = shortcut[:, :, :, :ch_out]
                shortcut2 = shortcut[:, :, :, ch_out:]
            else:
                shortcut1 = shortcut[:, :ch_out, :, :]
                shortcut2 = shortcut[:, ch_out:, :, :]
            shortcut2 = Conv2D('conv_short', shortcut2, ch_out, 1, strides=strides)
            shortcut2 = BatchNorm('bn_short', shortcut2)
            shortcut = shortcut1 + shortcut2
        l += shortcut
    return l

def residual_bottleneck_layer(name, l, out_filters, strides, data_format):
    data_format = get_data_format(data_format, keras_mode=False)
    ch_dim = 3 if data_format == 'NHWC' else 1
    ch_in = _get_dim(l, ch_dim)

    ch_base = out_filters
    ch_last = ch_base * 4
    l_in = l
    with tf.variable_scope('{}.0'.format(name)):
        l = BatchNorm('bn0', l)
        l = tf.nn.relu(l)
        l = (LinearWrap(l)
             .Conv2D('conv1x1_0', ch_base, 1, activation=BNReLU)
             .Conv2D('conv3x3_1', ch_base, 3, strides=strides, activation=BNReLU)
             .Conv2D('conv1x1_2', ch_last, 1)())
        l = BatchNorm('bn_3', l)

        shortcut = l_in
        if ch_in != ch_last:
            shortcut = Conv2D('conv_short', shortcut, ch_last, 1, strides=strides)
            shortcut = BatchNorm('bn_short', shortcut)
        l = l + shortcut
    return l

def initial_convolution(image, init_channel, s_type='basic', name='init_conv0'):
    with tf.variable_scope(name):
        if s_type == 'basic':
            l = Conv2D('conv0', image, init_channel, 3)
        elif s_type == 'imagenet':
            l = (LinearWrap(image)
                 .Conv2D('conv0', init_channel, 7, strides=2, activation=tf.identity)
                 .MaxPooling('pool0', 3, strides=2, padding='same')())
        elif s_type == 'conv7':
            l = Conv2D('conv0_7x7', image, init_channel, 7, strides=2)
        elif s_type == 'conv3':
            l = Conv2D('conv0_3x3', image, init_channel, 3, strides=2)
        else:
            raise Exception("Unknown starting type (s_type): {}".format(s_type))
        l = BatchNorm('init_bn', l)
    return l

def _reduce_prev_layer(p_layer, p_id, c_layer, out_filters, data_format, hw_only=False):
    """
    Make p_layer to have same shape as c_layer.
    """
    if p_layer is None:
        return c_layer
    c_ch = out_filters
    ch_dim = _data_format_to_ch_dim(data_format)
    tensor_dim = len(c_layer.get_shape().as_list())
    p_ch = _get_dim(p_layer, ch_dim)
    c_hw = None if tensor_dim == 2 else _get_dim(c_layer, 2)
    p_hw = None if tensor_dim == 2 else _get_dim(p_layer, 2)
    if p_hw != c_hw:
        # This implies they are not None and they are not the same size.
        if p_hw > c_hw:
            n_iter = int(np.log2(p_hw / c_hw))
            id_str = "reduce_{:03d}".format(p_id) if isinstance(p_id, int) else str(p_id)
            re_ret = re.search('{}_([0-9]*)'.format(id_str), p_layer.name)
            if re_ret is None:
                reduction_idx = 0
            else:
                reduction_idx = int(re_ret.group(1)) + 1
            out_filters = c_ch / (2 ** n_iter)
            for ridx in range(reduction_idx, reduction_idx + n_iter):
                out_filters *= 2
                p_layer = _factorized_reduction(
                    id_str + '_{}'.format(ridx),
                    p_layer, out_filters, data_format)
        else:
            raise NotImplementedError("Currently resizing/deconv up is not supported")
    elif c_ch != p_ch and not hw_only:
        p_layer = projection_layer('reduce_proj_{:03d}'.format(p_id), p_layer, c_ch, ch_dim)
    return p_layer

def _factorized_reduction(scope_name, x, out_filters, data_format):
    """Reduces the shape of x without information loss due to striding.
    copied from https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py
    """
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")

    #with tf.variable_scope(scope_name):
    #    layer = Conv2D('conv3x3_path', x, out_filters, 3, strides=2, activation=BNReLU)
    #return layer

    with tf.variable_scope(scope_name):
        path1 = AvgPooling('path1', x, pool_size=1, strides=2, padding='valid')
        path1 = Conv2D('path1_conv', path1, out_filters // 2, 1, padding='same')

        # Skip path 2
        # First pad with 0"s on the right and bottom, then shift the filter to
        # include those 0"s that were added.
        data_format = get_data_format(data_format, keras_mode=False)

        if data_format == "NHWC":
            pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
            path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
            ch_dim = 3
        else:
            pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
            path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
            ch_dim = 1

        path2 = AvgPooling('path2', x, pool_size=1, strides=2, padding='valid')
        path2 = Conv2D('path2_conv', path1, out_filters // 2, 1, padding='same')

        final_path = tf.concat(values=[path1, path2], axis=ch_dim)
        final_path = BatchNorm('bn', final_path)
        return final_path
