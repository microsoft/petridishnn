import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils import anytime_loss, logger, utils, fs

import bisect

from anytime_network import *



################################################
# FCN for semantic labeling
#
# NOTE:
#   Since the layer H/W are induced based on
#   the input layer rn, DO NOT give inputs of
#   H/W that are not divisible by 2**n_pools.
#   Otherwise the upsampling results in H/W that
#   mismatch the input H/W. (And screw dynamic
#   shape checking in tensorflow....)
################################################
def parser_add_fcn_arguments(parser):
    parser.add_argument('--is_label_one_hot',
        help='whether label img contains distribution of labels',
        default=False, action='store_true')
    parser.add_argument('--eval_threshold',
        help='The minimum valid labels weighting in [0,1] to trigger evaluation',
        type=np.float32, default=0.5)
    parser.add_argument('--n_pools',
        help='number of pooling blocks on the cfg.n_units_per_block',
        type=int, default=None)
    parser.add_argument('--do_scale_feat_to_label',
        help='Whether scale the feature map to the resolution of input image for evaluation',
        default=False, action='store_true')
    parser.add_argument('--input_height',
        default=None, type=int)
    parser.add_argument('--input_width',
        default=None, type=int)
    return parser


def atrous_pool_features(l, ch_out, atrous_rates, data_format='channels_first', scope_name='atrous_pool_feat'):
    input_shape = tf.shape(l)
    ch_dim = 1 if data_format=='channels_first' else 3
    h_dim = ch_dim % 3 + 1
    w_dim = h_dim + 1
    dyn_hw = [input_shape[h_dim], input_shape[w_dim]]

    with tf.variable_scope(scope_name) as scope:
        image_feat = GlobalAvgPooling('gap', l)
        image_feat_shape = [-1, 1, 1, 1]
        image_feat_shape[ch_dim] = l.get_shape().as_list()[ch_dim]
        image_feat = tf.reshape(image_feat, image_feat_shape)
        image_feat = Conv2D('gap_conv1x1', image_feat, ch_out, 1, activation=BNReLU)
        image_feat = ResizeImages('image_feat_resize', image_feat, dyn_hw)

        l_1x1 = Conv2D('conv1x1', l, ch_out, 1, activation=BNReLU)

        feats = [l_1x1, image_feat]

        for ar_i, rate in enumerate(atrous_rates):
            l_atrous = Conv2D('atrous_conv3x3_{}'.format(ar_i), l, ch_out, \
                3, dilation_rate=rate, activation=BNReLU)
            feats.append(l_atrous)

        l = tf.concat(feats, ch_dim, name='concat_feats')
        l = Conv2D('conv1x1_merge', l, ch_out, 1, activation=BNReLU)
    return l




class AnytimeFCN(AnytimeNetwork):
    """
        Overload AnytimeNetwork from classification set-up to semantic labeling:
        (1) the input now accept image and image labels;
        (2) prediction and loss are on image of prediction probs;
        (3) parse input label to various sizes of label distributions for ANN;
        (4) scene parsing callbacks.
        (5) TODO introduce transition_up and compute_ll_feats
    """

    def __init__(self, args):
        super(AnytimeFCN, self).__init__(None, args)


        # Class weight for fully convolutional networks
        self.class_weight = None
        if hasattr(args, 'class_weight'):
            self.class_weight = args.class_weight
        if self.class_weight is None:
            self.class_weight = np.ones(self.num_classes, dtype=np.float32)
        logger.info('Class weights: {}'.format(self.class_weight))

        self.is_label_one_hot = args.is_label_one_hot
        self.eval_threshold = args.eval_threshold
        self.do_scale_feat_to_label = args.do_scale_feat_to_label
        self.n_pools = args.n_pools if not self.do_scale_feat_to_label else 0
        self.is_atrous = args.is_atrous
        self.output_stride = args.output_stride
        # base_o_s / o_s * base_rate ; base_o_s == 16
        self.atrous_rates = [6,12,18]
        self.atrous_rates_base_output_stride = 16

        self.input_height = args.input_height
        self.input_width = args.input_width


    def compute_classification_callbacks(self):
        vcs = []
        total_units = self.total_units
        unit_idx = -1
        layer_idx=-1
        for n_units in self.network_config.n_units_per_block:
            for k in range(n_units):
                layer_idx += 1
                unit_idx += 1
                weight = self.weights[unit_idx]
                if weight > 0:
                    scope_name = self.compute_scope_basename(layer_idx)
                    scope_name = self.prediction_scope(scope_name) + '/'
                    vcs.append(MeanIoUFromConfusionMatrix(\
                        cm_name=scope_name+'confusion_matrix/SparseTensorDenseAdd:0',
                        scope_name_prefix=scope_name+'val_'))
                    vcs.append(WeightedTensorStats(\
                        names=[scope_name+'sum_abs_diff:0',
                            scope_name+'prob_sqr_err:0',
                            scope_name+'cross_entropy_loss:0'],
                        weight_name='dynamic_batch_size:0',
                        prefix='val_'))
        return vcs


    def inputs(self):
        HW = [self.input_height, self.input_width]
        if self.options.is_label_one_hot:
            # the label one-hot is in fact a distribution of labels.
            # Void labeled pixels have 0-vector distribution.
            label_desc = InputDesc(tf.float32,
                [None, HW[0], HW[1], self.num_classes], 'label')
        else:
            label_desc = InputDesc(tf.int32, [None, HW[0], HW[1]], 'label')
        return [InputDesc(self.input_type, [None, HW[0], HW[1], 3], 'input'), label_desc]


    def _parse_inputs(self, inputs):
        # NOTE
        # label_img is always NHWC/NHW/channel_last
        # If label_img is not one-hot, the distribution doesn't include void.
        # label_img is 0-vec for void labels
        image, label_img = inputs
        if not self.options.is_label_one_hot:
            # From now on label_img is tf.float one hot, void has 0-vector.
            # because we assume void >=num_classes
            label_img = tf.one_hot(label_img, self.num_classes, axis=-1)

        def nonvoid_mask(prob_img, name=None):
            mask = tf.cast(tf.greater(tf.reduce_sum(prob_img, axis=-1),
                                      self.options.eval_threshold),
                           dtype=tf.float32)
            mask = tf.reshape(mask, [-1], name=name)
            # TODO is this actually beneficial; and which KeepProb to use?
            #mask = Dropout(name, mask, keep_prob=0.5)
            return mask

        def flatten_label(prob_img, name=None):
            return tf.reshape(prob_img, [-1, self.num_classes], name=name)

        l_mask = []
        l_label = []
        l_dyn_hw = []
        label_img = tf.identity(label_img, name='label_img_0')
        n_label_scales = self.n_pools + 1 if not self.do_scale_feat_to_label else 1
        for pi in range(n_label_scales):
            l_mask.append(nonvoid_mask(label_img, 'eval_mask_{}'.format(pi)))
            l_label.append(flatten_label(label_img, 'label_{}'.format(pi)))
            img_shape = tf.shape(label_img)
            l_dyn_hw.append([img_shape[1], img_shape[2]])
            if pi == self.n_pools:
                break
            label_img = AvgPooling('label_img_{}'.format(pi+1), label_img, 2, \
                                   padding='same', data_format='channels_last')
        return image, [l_label, l_mask, l_dyn_hw]


    def bi_to_scale_idx(self, bi):
        ## Because there are n_pools number of pooling, there are n_pools+1 scales
        # Case downsample check: n_pools uses label_idx=n_pools;
        #   0 uses 0.
        # Case upsample check: bi == n_pools uses label_idx=n_pools;
        #   the final bi == n_pools * 2 uses 0
        if bi <= self.n_pools:
            return bi
        else:
            return 2*self.n_pools - bi

    def _compute_prediction_and_loss(self, l, label_inputs, unit_idx):
        l_label, l_eval_mask, l_dyn_hw = label_inputs

        ## Ground truth
        # compute block idx
        layer_idx = unit_idx
        # first idx that is > layer_idx
        bi = bisect.bisect_right(self.cumsum_blocks, layer_idx)
        label_img_idx = self.bi_to_scale_idx(bi) if not self.do_scale_feat_to_label else 0
        label = l_label[label_img_idx] # note this is a probability of label distri
        eval_mask = l_eval_mask[label_img_idx]
        dyn_hw = l_dyn_hw[label_img_idx]
        n_non_void_samples = tf.reduce_sum(eval_mask)
        n_non_void_samples += tf.cast(tf.less_equal(n_non_void_samples, 1e-12), tf.float32)

        ## Compute flattened logits
        # Assume all previous layers have gone through BNReLU, so conv directly
        ch_in = l.get_shape().as_list()[self.ch_dim]
        l = Conv2D('linear', l, self.num_classes, 1, use_bias=True)
        logit_vars = l.variables
        if self.data_format == 'channels_first':
            l = tf.transpose(l, [0,2,3,1])
        if self.do_scale_feat_to_label:
            # at this stage, the logits are already channels_last
            l = ResizeImages('resize_logits', l, dyn_hw, data_format='channels_last')
        logits = tf.reshape(l, [-1, self.num_classes], name='logits')
        variables = [logit_vars.W, logit_vars.b]

        ## Square error between distributions.
        # Implement our own here b/c class weighting.
        prob = tf.nn.softmax(logits, name='pred_prob')
        prob_img_shape = tf.stack([-1,  dyn_hw[0], dyn_hw[1], self.num_classes])
        prob_img = tf.reshape(prob, prob_img_shape, name='pred_prob_img')
        sqr_err = tf.reduce_sum(\
            tf.multiply(tf.square(label - prob), self.class_weight), \
            axis=1, name='pixel_prob_square_err')
        sqr_err = tf.divide(tf.reduce_sum(sqr_err * eval_mask), n_non_void_samples,
            name='prob_sqr_err')
        add_moving_summary(sqr_err)

        ## Weighted cross entropy
        # Have to implement our own weighted softmax cross entroy
        # because TF doesn't provide one
        # Because logits and cost are returned in the end of this func,
        # we use _logit to represent  the shifted logits.
        max_logits = tf.reduce_max(logits, axis=1, keep_dims=True)
        _logits = logits - max_logits
        normalizers = tf.reduce_sum(tf.exp(_logits), axis=1, keep_dims=True)
        _logits = _logits - tf.log(normalizers)
        cross_entropy = -tf.reduce_sum(\
            tf.multiply(label * _logits, self.class_weight), axis=1)
        cross_entropy = cross_entropy * eval_mask
        cross_entropy = tf.divide(tf.reduce_sum(cross_entropy), n_non_void_samples,
                                  name='cross_entropy_loss')
        add_moving_summary(cross_entropy)

        ## Unweighted total abs diff
        sum_abs_diff = sum_absolute_difference(prob, label)
        sum_abs_diff *= eval_mask
        sum_abs_diff = tf.divide(tf.reduce_sum(sum_abs_diff),
                                 n_non_void_samples, name='sum_abs_diff')
        add_moving_summary(sum_abs_diff)

        ## confusion matrix for iou and pixel level accuracy
        int_pred = tf.argmax(logits, 1, name='int_pred')
        int_label = tf.argmax(label, 1, name='int_label')
        cm = tf.confusion_matrix(labels=int_label, predictions=int_pred,\
            num_classes=self.num_classes, name='confusion_matrix', weights=eval_mask)

        ## pixel level accuracy
        accu = tf.divide(tf.cast(tf.reduce_sum(tf.diag_part(cm)), dtype=tf.float32), \
                         n_non_void_samples, name='accuracy')
        add_moving_summary(accu)
        return cross_entropy, variables


    def _init_extra_info(self):
        return None

    def _compute_block_and_transition(self, pls, pmls, n_units, bi, extra_info):
        return pls, pmls

    def _compute_ll_feats(self, image):
        l_feats = self._compute_init_l_feats(image)
        pls = [l_feats[0]]
        pmls = []
        extra_info = self._init_extra_info()
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            pls, pmls, extra_info = self._compute_block_and_transition(\
                pls, pmls, n_units, bi, extra_info)

        ll_feats = [ [ feat ] for feat in pmls ]
        assert len(ll_feats) == self.total_units
        return ll_feats


###########
## FCDense
# This class reproduces tiramisu FC-Densenet for scene parsing.
# It is meant for a comparison against anytime FCN
def parser_add_fcdense_arguments(parser):
    parser = parser_add_fcn_arguments(parser)
    parser, depth_group = parser_add_densenet_arguments(parser)
    depth_group.add_argument('--fcdense_depth',
                            help='depth of the network in number of conv',
                            type=int)
    parser.add_argument('--is_atrous', help='whether use atrous conv or using deconv for fine resolution',
        default=False, action='store_true')
    parser.add_argument('--output_stride', help='Under is_atrous, ' \
        +' the ratio between input img to final feature resolution',
        type=int, default=16, choices=[8,16,32])
    return parser, depth_group


## FCDense
# This class reproduces tiramisu FC-Densenet for scene parsing.
# It is meant for a comparison against anytime FCN
class FCDensenet(AnytimeFCN):
    def __init__(self, args):
        super(FCDensenet, self).__init__(args)
        self.reduction_ratio = self.options.reduction_ratio
        self.growth_rate = self.options.growth_rate

        # Class weight for fully convolutional networks
        self.class_weight = None
        if hasattr(args, 'class_weight'):
            self.class_weight = args.class_weight
        if self.class_weight is None:
            self.class_weight = np.ones(self.num_classes, dtype=np.float32)
        logger.info('Class weights: {}'.format(self.class_weight))

        # other format is not supported yet
        assert self.n_pools * 2 + 1 == self.n_blocks

        # FC-dense doesn't like the starting pooling of imagenet initial conv/pool
        assert self.network_config.s_type == 'basic'

        # TODO This version doesn't support anytime prediction (yet)
        assert self.options.func_type == FUNC_TYPE_OPT

    def _transition_up(self, skip_stack, l_layers, bi):
        with tf.variable_scope('TU_{}'.format(bi)) as scope:
            stack = tf.concat(l_layers, self.ch_dim, name='concat_recent')
            ch_out = stack.get_shape().as_list()[self.ch_dim]
            dyn_hw = [tf.shape(skip_stack)[self.h_dim], tf.shape(skip_stack)[self.w_dim]]
            stack = Deconv2D('deconv', stack, ch_out, 3, strides=2)
            stack = ResizeImages('resize', stack, [dyn_h, dyn_w])
            stack = tf.concat([skip_stack, stack], self.ch_dim, name='concat_skip')
        return stack

    def _transition_down(self, stack, bi):
        with tf.variable_scope('TD_{}'.format(bi)) as scope:
            stack = BNReLU('bnrelu', stack)
            ch_in = stack.get_shape().as_list()[self.ch_dim]
            stack = Conv2D('conv1x1', stack, ch_in, 1, use_bias=True)
            stack = Dropout('dropout', stack, keep_prob=self.dropout_kp)
            stack = MaxPooling('pool', stack, 2, padding='same')
        return stack

    def _dense_block(self, stack, n_units, init_uidx):
        unit_idx = init_uidx
        l_layers = []
        for ui in range(n_units):
            unit_idx += 1
            scope_name = self.compute_scope_basename(unit_idx)
            with tf.variable_scope(scope_name+'.feat'):
                stack = BNReLU('bnrelu', stack)
                l = Conv2D('conv3x3', stack, self.growth_rate, 3, use_bias=True)
                if self.dropout_kp < 1:
                    l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                stack = tf.concat([stack, l], self.ch_dim, name='concat_feat')
                l_layers.append(l)
        return stack, unit_idx, l_layers

    def _compute_ll_feats(self, image):
        # compute init feature
        l_feats = self._compute_init_l_feats(image)
        stack = l_feats[0]

        n_units_per_block = self.network_config.n_units_per_block

        # compute downsample blocks
        unit_idx = -1
        skip_connects = []
        for bi in range(self.n_pools):
            n_units = n_units_per_block[bi]
            stack, unit_idx, l_layers = self._dense_block(stack, n_units, unit_idx)
            skip_connects.append(stack)
            stack = self._transition_down(stack, bi)

        # center block
        skip_connects = list(reversed(skip_connects))
        n_units = n_units_per_block[self.n_pools]
        stack, unit_idx, l_layers = self._dense_block(stack, n_units, unit_idx)

        # upsampling blocks
        for bi in range(self.n_pools):
            stack = self._transition_up(skip_connects[bi], l_layers, bi)
            n_units = n_units_per_block[bi+self.n_pools+1]
            stack, unit_idx, l_layers = self._dense_block(stack, n_units, unit_idx)

        # interface with AnytimeFCN
        # creat dummy feature for previous layers, and use stack as final feat
        ll_feats = [ [None] for i in range(unit_idx+1) ]
        stack = BNReLU('bnrelu_final', stack)
        ll_feats[-1] = [stack]
        return ll_feats


def AnytimeFCDenseNet(T_class):

    class AnytimeFCDenseNetTemplate(AnytimeFCN, T_class):
        """
            Anytime FC-densenet. Use AnytimeFCN to have FC input, logits, costs.

            Use T_class to have pre_compute_connections, compute_transition,
            compute_block

        """

        def __init__(self, args):
            # set up params from regular densenet.
            super(AnytimeFCDenseNetTemplate, self).__init__(args)
            self.dropout_kp = 1.0

            # other format is not supported yet
            assert self.n_pools * 2 + 1 == self.n_blocks
            # FC-dense doesn't like the starting pooling of imagenet initial conv/pool
            assert self.network_config.s_type == 'basic'


        def compute_block(self, pls, pmls, n_units, growth, max_merge):
            """
                pls : previous layers. including the init_feat. Hence pls[i] is from
                    layer i-1 for i > 0
                pmls : previous merged layers. (used for generate ll_feats)
                n_units : num units in a block
                max_merge : upper bound for how many selected layers can be merged at once,
                    if we select more than max_merge, we will have multiple batches of size
                    max_merge. Each batch will produce a result, and the results will be added
                    together.  This is for preventing the 2GB max tensor size problem

                return pls, pmpls (updated version of these)
            """
            unit_idx = len(pls) - 2 # unit idx of the last completed unit
            for _ in range(n_units):
                unit_idx += 1
                scope_name = self.compute_scope_basename(unit_idx)
                with tf.variable_scope(scope_name+'.feat'):
                    sl_indices = self.connections[unit_idx]
                    logger.info("unit_idx = {}, len past_feats = {}, selected_feats: {}".format(\
                        unit_idx, len(pls), sl_indices))

                    for idx_st in range(0, len(sl_indices), max_merge):
                        idx_ed = min(idx_st + max_merge, len(sl_indices))
                        name_appendix = '' if idx_st == 0 else '_{}'.format(idx_st)
                        ml = tf.concat([pls[sli] for sli in sl_indices[idx_st:idx_ed]],\
                                       self.ch_dim, name='concat_feat'+name_appendix)

                        # pre activation
                        l = BNReLU('bnrelu_merged'+name_appendix, ml)

                        # First conv
                        layer_growth = self._compute_layer_growth(unit_idx, growth)
                        if self.network_config.b_type == 'bottleneck':
                            bottleneck_width = int(self.options.bottleneck_width * layer_growth)
                            l = Conv2D('conv1x1'+name_appendix, l, bottleneck_width, 1, activation=BNReLU)
                        else:
                            l = Conv2D('conv3x3'+name_appendix, l, layer_growth, 3)
                        l = Dropout('dropout', l, keep_prob=self.dropout_kp)

                        # accumulate conv results
                        if idx_st == 0:
                            l_sum = l
                        else:
                            l_sum += l

                    # for bottleneck case, 2nd conv using the accumulated result
                    l = l_sum
                    if self.network_config.b_type == 'bottleneck':
                        l = Conv2D('conv3x3'+name_appendix, l, layer_growth, 3)
                        l = Dropout('dropout', l, keep_prob=self.dropout_kp)
                    pls.append(l)

                    # If the feature is used for prediction, store it.
                    if self.weights[unit_idx] > 0:
                        # TODO if there are more than max merge layers, this will cause error
                        pred_feat = [pls[sli] for sli in sl_indices] + [l]
                        pmls.append(tf.concat(pred_feat, self.ch_dim, name='concat_pred'))
                    else:
                        pmls.append(None)
            return pls, pmls


        def _compute_transition_up(self, pls, skip_pls, trans_idx):
            """ for each previous layer, transition it up with deconv2d i.e.,
                conv2d_transpose
            """
            ## current scale
            scale_bi = self.bi_to_scale_idx(trans_idx)
            new_pls = []
            for pli, pl in enumerate(pls):
                if pli < len(skip_pls):
                    new_pls.append(skip_pls[pli])
                    continue
                if self.l_min_scale[pli] >= scale_bi:
                    new_pls.append(None)
                    continue
                # implied that skip_pls is exhausted
                with tf.variable_scope('transit_{:02d}_{:02d}'.format(trans_idx, pli)):
                    ch_in = pl.get_shape().as_list()[self.ch_dim]
                    ch_out = int(ch_in * self.reduction_ratio)
                    shapes = tf.shape(skip_pls[0])
                    dyn_hw = [shapes[self.h_dim], shapes[self.w_dim]]
                    pl = BNReLU('pre_bnrelu', pl)
                    pl = Deconv2D('deconv', pl, ch_out, 3, strides=2)
                    pl = ResizeImages('resize', pl, dyn_hw)
                    new_pls.append(pl)
            return new_pls


        ## Extra info for anytime FC (log)DenseNets:
        # growth (float32) : which can technically change over time
        # l_pls  (list) : store the list of pls at different scales.
        def _init_extra_info(self):
            ## fill up l_min/max scale
            self.pre_compute_connections()
            extra_info = (self.growth_rate, [])
            return extra_info


        def _compute_block_and_transition(self, pls, pmls, n_units, bi, extra_info):
            growth, l_pls = extra_info
            max_merge = 4096
            if bi >= self.n_blocks -2:
                max_merge = 8
            pls, pmls = self.compute_block(pls, pmls, n_units, growth, max_merge)
            if bi < self.n_pools:
                l_pls.append(pls)
                growth *= self.growth_rate_multiplier
                pls = self.compute_transition(pls, bi)
            elif bi < self.n_blocks - 1:
                growth /= self.growth_rate_multiplier
                skip_pls = l_pls[self.bi_to_scale_idx(bi) - 1]
                pls = self._compute_transition_up(pls, skip_pls, bi)
            return pls, pmls, (growth, l_pls)

    return AnytimeFCDenseNetTemplate


## Version 2 of anytime FCN for dense-net
#
class AnytimeFCDenseNetV2(AnytimeFCN, AnytimeLogDenseNetV2):

    def __init__(self, args):
        super(AnytimeFCDenseNetV2, self).__init__(args)
        assert self.n_pools * 2 == self.n_blocks - 1
        assert self.network_config.s_type == 'basic'

    def update_compressed_feature_up(self, layer_idx, ch_out, pls, bcml, sml):
        """
            layer_idx :
            pls : list of layers in the most recent block
            bcml : context feature for the most recent block
            sml : Final feature of the target scale on the down path
                (it comes from skip connection)
        """
        with tf.variable_scope('transition_after_{}'.format(layer_idx)) as scope:
            l = tf.concat(pls, self.ch_dim, name='concat_new')
            shapes = tf.shape(sml)
            dyn_hw = [shapes[self.h_dim], shapes[self.w_dim]]
            l = BNReLU('pre_bnrelu', l)
            l = Deconv2D('deconv_new', l, ch_out, 3, strides=2)
            l = ResizeImages('resize_new', l, dyn_hw)

            bcml = BNReLU('pre_bnrelu_bcml', bcml)
            bcml = Deconv2D('deconv_old', bcml, ch_out, 3, strides=2)
            bcml = ResizeImages('resize_old', bcml, dyn_hw)
            bcml = tf.concat([sml, l, bcml], self.ch_dim, name='concat_all')
        return bcml

    def _compute_ll_feats(self, image):
        self.pre_compute_connections()
        l_feats = self._compute_init_l_feats(image)
        bcml = l_feats[0] #block compression merged layer
        l_mls = []
        growth = self.growth_rate
        layer_idx = -1
        l_skips = []
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            pls, l_mls = self.compute_block(layer_idx, n_units, l_mls, bcml, growth)
            layer_idx += n_units
            ch_out = growth * (int(np.log2(self.total_units + 1)) + 1)
            if bi < self.n_pools:
                l_skips.append(l_mls[-1])
                bcml = self.update_compressed_feature(layer_idx, ch_out, pls, bcml)
            elif bi < self.n_blocks - 1:
                sml = l_skips[self.bi_to_scale_idx(bi) - 1]
                bcml = self.update_compressed_feature_up(layer_idx, ch_out, pls, bcml, sml)

        ll_feats = [ [ BNReLU('bnrelu_{}'.format(li), ml) ] if self.weights[li] > 0 else [None]
            for li, ml in enumerate(l_mls) ]
        assert len(ll_feats) == self.total_units
        return ll_feats


#####################
# Coarse to Fine FCN
#####################
class AnytimeFCNCoarseToFine(AnytimeFCN):
    def __init__(self, args):
        super(AnytimeFCNCoarseToFine, self).__init__(args)
        self.growth_rate = self.options.growth_rate
        self.growth_rate_factor = [1,2,4,4]
        self.bottleneck_factor = [1,2,4,4]
        self.output_strides = [4, 8, 16, 32]
        if not self.is_atrous:
            self.output_stride = 32
        self.output_stride_ratio = [ (o_stride // self.output_stride) for o_stride in self.output_strides ]

        self.init_channel = self.growth_rate * 2
        self.reduction_ratio = self.options.reduction_ratio
        self.num_scales = len(self.network_config.n_units_per_block)
        self.n_pools = self.num_scales - 1 if not self.do_scale_feat_to_label else 0
        self.dropout_kp = args.dropout_kp


    def bi_to_scale_idx(self, bi):
        return bi


    def _compute_edge(self, l, ch_out, bnw, osr,
            l_type='normal', dyn_hw=None,
            name=""):
        if self.network_config.b_type == 'bottleneck':
            bnw_ch = int(bnw * ch_out)
            l = Conv2D('conv1x1_'+name, l, bnw_ch, 1, activation=BNReLU)
        dilation_rate = 1
        stride = 1
        if l_type == 'normal':
            dilation_rate = max(1, osr)
        elif l_type == 'down':
            stride = 1 if osr > 1 else 2
            dilation_rate = max(1, osr//2)
        elif l_type == 'up':
            assert dyn_hw is not None
            l = ResizeImages('resize', l, dyn_hw)
            #dilation_rate = max(1, osr//2)
        l = Conv2D('conv3x3_'+name, l, ch_out, 3, strides=stride, \
            dilation_rate=dilation_rate, activation=BNReLU)
        return l

    def _compute_block(self, bi, n_units, layer_idx, l_mf):
        ll_feats = []
        for k in range(n_units):
            layer_idx += 1
            scope_name = self.compute_scope_basename(layer_idx)
            s_start = self.bi_to_scale_idx(bi)
            s_end = self.num_scales
            l_feats = [None] * self.num_scales
            for w in range(s_start, s_end):
                with tf.variable_scope(scope_name+'.'+str(w)) as scope:
                    g = self.growth_rate_factor[w] * self.growth_rate
                    bnw = self.bottleneck_factor[w]
                    osr = self.output_stride_ratio[w]
                    has_prev_scale = w > 0 and l_mf[w-1] is not None
                    has_next_scale = w < self.num_scales - 1 and l_mf[w+1] is not None
                    dyn_hw = [tf.shape(l_mf[w])[self.h_dim], tf.shape(l_mf[w])[self.w_dim]]

                    if has_prev_scale:
                        l = self._compute_edge(l_mf[w], g//2, bnw, osr, 'normal', name='e1')
                        bnw_prev = self.bottleneck_factor[w-1]
                        lp = self._compute_edge(l_mf[w-1], g - g//2, bnw_prev, osr, 'down', name='e2')
                        l = tf.concat([l, lp], self.ch_dim, name='concat_ms')
                    else:
                        l = self._compute_edge(l_mf[w], g, bnw, osr, 'normal')
                    l_feats[w] = l

            #end for w
            new_l_mf = [None] * self.num_scales
            for w in range(s_start, s_end):
                with tf.variable_scope(scope_name+'.'+str(w) + '.merge') as scope:
                    new_l_mf[w] = tf.concat([l_mf[w], l_feats[w]], self.ch_dim, name='merge_feats')
            ll_feats.append(new_l_mf)
            l_mf = new_l_mf
        return ll_feats


    def _compute_transition(self, l_mf, layer_idx):
        rr = self.reduction_ratio
        l_feats = [None] * self.num_scales
        for w, l in enumerate(l_mf):
            if l is None:
                continue

            with tf.variable_scope('transat_{:02d}_{:02d}'.format(layer_idx, w)):
                ch_in = l.get_shape().as_list()[self.ch_dim]
                ch_out = int(ch_in * rr)
                l = Conv2D('conv1x1', l, ch_out, 1, activation=BNReLU)
                l_feats[w] = l
        return l_feats


    def _compute_init_l_feats(self, image):
        l_feats = []
        for w in range(self.num_scales):
            scope_name = self.compute_scope_basename(0)
            with tf.variable_scope(scope_name+'.'+str(w)) as scope:
                ch_out = self.init_channel * self.growth_rate_factor[w]
                if w == 0:
                    if self.network_config.s_type == 'basic':
                        l = Conv2D('conv3x3', image, ch_out, 3, activation=BNReLU)
                    else:
                        assert self.network_config.s_type == 'imagenet'
                        l = (LinearWrap(image)
                            .Conv2D('conv7x7', ch_out, 7, strides=2, activation=BNReLU)
                            .MaxPooling('pool0', 3, strides=2, padding='same')())
                else:
                    osr = self.output_stride_ratio[w]
                    stride = 1 if osr > 1 else 2
                    dilation_rate = max(1, osr//2)
                    l = Conv2D('conv3x3', l, ch_out, 3, strides=stride, \
                        dilation_rate=dilation_rate, activation=BNReLU)
                l_feats.append(l)
        return l_feats


    def _compute_ll_feats(self, image):
        l_mf = self._compute_init_l_feats(image)
        ll_feats = [ l_mf ]
        layer_idx = 0
        for bi, n_units in enumerate(self.network_config.n_units_per_block):
            ll_block_feats = self._compute_block(bi, n_units - 1, layer_idx, l_mf)
            layer_idx += n_units - 1
            ll_feats.extend(ll_block_feats)
            l_mf = ll_block_feats[-1]
            if bi < self.n_blocks - 1 and self.reduction_ratio < 1:
                l_mf = self._compute_transition(l_mf, layer_idx)
                ll_feats.append(l_mf)
                layer_idx += 1

        # force merge with the last of the first block.
        n_first_block_layers = self.network_config.n_units_per_block[0]
        fine_feat = ll_feats[n_first_block_layers - 1][0]
        fine_shape = tf.shape(fine_feat)
        fine_hw = [fine_shape[self.h_dim], fine_shape[self.w_dim]]

        ll_feats_ret = [None] * self.total_units
        for li, l_feats in enumerate(ll_feats):
            if self.weights[li] == 0:
                continue

            scope_name = self.compute_scope_basename(li)
            with tf.variable_scope(scope_name + '.pred_feat') as scope:
                bi = bisect.bisect_right(self.cumsum_blocks, li)
                si = self.bi_to_scale_idx(bi)

                l = l_feats[-1]
                ch_out = 256
                atrous_rates = np.asarray(self.atrous_rates)
                l = atrous_pool_features(l, ch_out, atrous_rates, data_format=self.data_format)

                if li < n_first_block_layers:
                    lp = l_feats[0]
                else:
                    lp = fine_feat
                lp = Conv2D('fine_conv1x1', lp, 48, 1, activation=BNReLU)

                l = ResizeImages('resize_to_decoder', l, fine_hw)
                l = tf.concat([l, lp], self.ch_dim, name='merge_with_fine_feat')
                l = Conv2D('decoder_conv3x3_1', l, ch_out, 3, activation=BNReLU)
                l = Conv2D('decoder_conv3x3_2', l, ch_out, 3, activation=BNReLU)
                ll_feats_ret[li] = [l]
        return ll_feats_ret


def sum_absolute_difference(pred, label, name='sum_abs_diff_by_dp'):
    """
    Args:
        pred: shape [B, C]
        label: shape [B, C]

        both pred and label need to be in the same space, 
        e.g., logit space, prob space, and etc.
    Return:
        sum of the absolute value of the difference of pred and label
        for each sample
    """
    return tf.reduce_sum(tf.abs(pred - label), axis=1)


