# -*- coding: utf-8 -*-
# File: conv2d.py


from ..compat import tfv1 as tf  # this should be avoided first in model code

from ..tfutils.common import get_tf_version_tuple
from ..utils.argtools import get_data_format, shape2d, shape4d, log_once
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['Conv2D', 'Deconv2D', 'Conv2DTranspose', 'GroupedConv2D', 'ResizeImages', 'SeparableConv2D']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1):
    """
    A wrapper around `tf.layers.Conv2D`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv. Note that this is not efficient.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')
    dilation_rate = shape2d(dilation_rate)

    if split == 1 and dilation_rate == [1, 1]:
        # tf.layers.Conv2D has bugs with dilations (https://github.com/tensorflow/tensorflow/issues/26797)
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

        # compute the flops of the conv
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'channels_last' else 1
        h_dim = 1 if data_format == 'channels_last' else 2
        w_dim = h_dim + 1
        in_channel = in_shape[channel_axis]
        out_channel = filters
        kernel_shape = shape2d(kernel_size)
        stride = shape4d(strides, data_format=data_format)
        flops = 1.0 * in_channel * out_channel * \
            kernel_shape[0] * kernel_shape[1] / stride[h_dim] / stride[w_dim]
        if in_shape[h_dim] is not None and in_shape[h_dim] > 0:
            flops *= in_shape[h_dim] * in_shape[w_dim]
        ret.info = VariableHolder(flops=flops)

    else:
        # group conv implementation
        data_format = get_data_format(data_format, keras_mode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = -1 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv now!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == [1, 1] or get_tf_version_tuple() >= (1, 5), 'TF>=1.5 required for dilated conv.'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_tuple() >= (1, 5):
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        conv = None
        if get_tf_version_tuple() >= (1, 13):
            try:
                conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
            except ValueError:
                conv = None
                log_once("CUDNN group convolution support is only available with "
                         "https://github.com/tensorflow/tensorflow/pull/25818 . "
                         "Will fall back to a loop-based slow implementation instead!", 'warn')
        if conv is None:
            inputs = tf.split(inputs, split, channel_axis)
            kernels = tf.split(W, split, 3)
            outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
                       for i, k in zip(inputs, kernels)]
            conv = tf.concat(outputs, channel_axis)

        if activation is None:
            activation = tf.identity
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b

        h_dim = 1 if data_format == 'NHWC' else 2
        w_dim = h_dim + 1
        flops = 1.0 * in_channel * out_channel * \
            kernel_shape[0] * kernel_shape[1] / stride[h_dim] / stride[w_dim] / split
        if in_shape[h_dim] is not None and in_shape[h_dim] > 0:
            flops *= in_shape[h_dim] * in_shape[w_dim]
        ret.info = VariableHolder(flops=flops)
    return ret


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2DTranspose(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        dyn_hw=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

    if get_tf_version_tuple() <= (1, 12):
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')
        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias
    else:
        # Our own implementation, to avoid Keras bugs. https://github.com/tensorflow/tensorflow/issues/25946
        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Unsupported arguments due to Keras bug in TensorFlow 1.13"
        data_format = get_data_format(data_format, keras_mode=False)
        shape_dyn = tf.shape(inputs)
        strides2d = shape2d(strides)
        channels_in = inputs.shape[1 if data_format == 'NCHW' else 3]
        if data_format == 'NCHW':
            channels_in = inputs.shape[1]
            out_shape_dyn = tf.stack(
                [shape_dyn[0], filters,
                 shape_dyn[2] * strides2d[0],
                 shape_dyn[3] * strides2d[1]])
            out_shape3_sta = [filters,
                              None if inputs.shape[2] is None else inputs.shape[2] * strides2d[0],
                              None if inputs.shape[3] is None else inputs.shape[3] * strides2d[1]]
        else:
            channels_in = inputs.shape[-1]
            out_shape_dyn = tf.stack(
                [shape_dyn[0],
                 shape_dyn[1] * strides2d[0],
                 shape_dyn[2] * strides2d[1],
                 filters])
            out_shape3_sta = [None if inputs.shape[1] is None else inputs.shape[1] * strides2d[0],
                              None if inputs.shape[2] is None else inputs.shape[2] * strides2d[1],
                              filters]

        kernel_shape = shape2d(kernel_size)
        W = tf.get_variable('W', kernel_shape + [filters, channels_in], initializer=kernel_initializer)
        if use_bias:
            b = tf.get_variable('b', [filters], initializer=bias_initializer)
        conv = tf.nn.conv2d_transpose(
            inputs, W, out_shape_dyn,
            shape4d(strides, data_format=data_format),
            padding=padding.upper(),
            data_format=data_format)
        conv.set_shape(tf.TensorShape([None] + out_shape3_sta))
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret


Deconv2D = Conv2DTranspose


@layer_register(log_shape=True)
def ResizeImages(
    images,
    size,
    method=tf.image.ResizeMethod.BILINEAR,
    align_corners=True,
    data_format='channels_last'):
    """
        Use tf.image.resize_images to resize feature map.
        We have to do some transposing first before using image resize if the
        data_format is 'channels_first', because resize_images only accept
        'channels_last'.

        images : tensor representing the feature map to resize
        size : 2D int32 tensor of the new shape (h, w)
        method : Resize method see tf.image.ResizeMethod
        align_corners : Preserving the corner pixels ?
        data_format : current data_format of the inputs 'channels_first' or 'channels_last'
    """
    l = images
    if data_format == 'channels_first':
        l = tf.transpose(l, [0,2,3,1])
    l = tf.image.resize_images(l, size, method, align_corners)
    if data_format == 'channels_first':
        l = tf.transpose(l, [0,3,1,2])
    ret = tf.identity(l, name='output')
    return ret


@layer_register(log_shape=True)
def GroupedConv2D(x, num_paths, path_ch_out, kernel_shape,
        sum_paths=False, padding='SAME', stride=1,
        W_init=None, b_init=None, nl=tf.identity,
        use_bias=False, data_format='NHWC'):
    """
    Grouped conv 2d for ResNeXt. Uses depthwise conv 2d and reshape and sum.

    Args:
        x : 4D tensor of data_format
        num_paths : number of groups
        path_ch_out : number of ch_out per group
        kernel_shape : (h,w) tuple or an int
        sum_paths : whether the groups are summed together (if True)
            or concatenated (if False (default))
        padding, W_init, b_init, nl, use_bias, data_format : see Conv2D

    Returns:
        tf.Tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    data_format = get_data_format(data_format, tfmode=False)

    in_shape = x.get_shape().as_list()
    ch_dim = 3 if data_format == 'NHWC' else 1
    ch_in = in_shape[ch_dim]
    assert ch_in % num_paths == 0, "Grouped conv requires n_groups to divide ch_in"
    ch_in_per_path = ch_in // num_paths
    ch_out = path_ch_out if sum_paths else num_paths * path_ch_out

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [ch_in, path_ch_out]
    stride = shape4d(stride, data_format=data_format)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer(2.0)
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [ch_out], initializer=b_init)

    x = tf.nn.depthwise_conv2d(x, W, stride, padding, rate=None, data_format=data_format)
    out_shape = x.get_shape().as_list()

    # First reshape to expose the dimension by input channels
    shape_depthwise = [num_paths, ch_in_per_path, path_ch_out]
    if data_format == 'NHWC':
        x = tf.reshape(x, [-1, out_shape[1], out_shape[2]] + shape_depthwise)
    else:
        x = tf.reshape(x, [-1] + shape_depthwise + [out_shape[2], out_shape[3]])

    # Then reduce sum to remove the input channel leaving output dim and (path dim)
    if sum_paths:
        sum_axis = [ch_dim, ch_dim + 1]
    else:
        sum_axis = ch_dim + 1
    x = tf.reduce_sum(x, sum_axis)

    # reshape to output shape if path dim did not collapse
    if not sum_paths:
        if data_format == 'NHWC':
            x = tf.reshape(x, [-1, out_shape[1], out_shape[2], ch_out])
        else:
            x = tf.reshape(x, [-1, ch_out, out_shape[2], out_shape[3]])

    ret = nl(tf.nn.bias_add(x, b, data_format=data_format) if use_bias else x, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret



@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def SeparableConv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1,1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1,1),
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        depthwise_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        pointwise_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    with rename_get_variable({'depthwise_kernel': 'Wd', 'pointwise_kernel': 'Wp', 'bias': 'b'}):
        layer = tf.layers.SeparableConv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depth_multiplier=depth_multiplier,
            depthwise_initializer=depthwise_initializer,
            pointwise_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            activity_regularizer=activity_regularizer,
            depthwise_regularizer=depthwise_regularizer,
            pointwise_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(Wd=layer.depthwise_kernel, Wp=layer.pointwise_kernel)
    if use_bias:
        ret.variables.b = layer.bias

    # compute the flops of the conv
    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'channels_last' else 1
    h_dim = 1 if data_format == 'channels_last' else 2
    w_dim = h_dim + 1
    in_channel = in_shape[channel_axis]
    out_channel = filters
    kernel_shape = shape2d(kernel_size)
    stride = shape4d(strides, data_format=data_format)
    flops = 1.0 * in_channel * depth_multiplier * \
        kernel_shape[0] * kernel_shape[1] / stride[h_dim] / stride[w_dim]
    # since pointwise is on the result of depthwise, the stride is carried over.
    pt_flops = 1.0 * (in_channel * depth_multiplier) * out_channel \
        / stride[h_dim] / stride[w_dim]
    if in_shape[h_dim] is not None and in_shape[h_dim] > 0:
        H_times_W = in_shape[h_dim] * in_shape[w_dim]
        flops *= H_times_W
        pt_flops *= H_times_W
    ret.info = VariableHolder(flops=flops+pt_flops)
    return ret
