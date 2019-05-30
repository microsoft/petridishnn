# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_global_step_var

def _drop_path(x, keep_prob):
    """
    Drops out a whole example hiddenstate with
    the specified probability.
    """
    batch_size = tf.shape(x)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    x = tf.div(x, keep_prob) * binary_tensor
    return x

def _apply_drop_path(
        x, drop_path_keep_prob, curr_depth, total_depth, max_train_steps):
    layer_ratio = float(curr_depth + 1) / total_depth
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

    curr_step = tf.to_float(get_global_step_var() + 1)
    step_ratio = curr_step / tf.to_float(max_train_steps)
    step_ratio = tf.minimum(1.0, step_ratio)
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    #with tf.device('/cpu:0'):
    #    tf.summary.scalar('layer_ratio', layer_ratio)
    #    tf.summary.scalar('step_ratio', step_ratio)
    x = _drop_path(x, drop_path_keep_prob)
    return x

class DropPath(object):

    def __init__(self, drop_path_keep_prob, max_train_steps, total_depth):
        self.max_train_steps = max_train_steps
        self.total_depth = total_depth
        self.is_training = get_current_tower_context().is_training
        self.drop_path_keep_prob = drop_path_keep_prob
        self.do_drop_path = (
            self.is_training and
            self.drop_path_keep_prob is not None and
            self.drop_path_keep_prob < 1.0
        )

    def __call__(self, x, curr_depth):
        """
        Args:
        x (Tensor) : Tensor to apply drop path on
        curr_depth (int) : non_input_layer_idx
        """
        if self.do_drop_path:
            return _apply_drop_path(
                x=x,
                drop_path_keep_prob=self.drop_path_keep_prob,
                curr_depth=curr_depth,
                total_depth=self.total_depth,
                max_train_steps=self.max_train_steps
            )
        return x
