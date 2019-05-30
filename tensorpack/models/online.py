import tensorflow as tf
import numpy as np

from ..tfutils.summary import *
from ..tfutils.tower import get_current_tower_context
from ..utils import logger

__all__ = ['Exp3', 'HalfEndHalfExp3', 'RandSelect', 'RWM']

class Exp3(object):
    def __init__(self, name, K, gamma):
        with tf.variable_scope(name) as scope:
            ctx = get_current_tower_context()
            self.inactive = ctx.is_training is not None and not ctx.is_training
            if self.inactive:
                return
            self.name = name
            self.K = K
            self.gamma = gamma
            self.w = tf.get_variable("w", [1, self.K], 
                initializer=tf.constant_initializer(1.0/self.K), trainable=False)
            for k in range(self.K):
                add_moving_summary(self.w[0][k])

    def sample(self):
        if self.inactive:
            return -1, 0.0

        with tf.variable_scope(self.name) as scope:
            scope.reuse_variables()
            w = tf.get_variable("w")
            probs = (1.0 - self.gamma) * w + self.gamma / self.K
            #w = tf.Print(w, [w], "sample")
            idx = tf.cast(tf.multinomial(tf.log(probs), 1)[0][0], tf.int32)
            p_idx = probs[0][idx]
            return idx, p_idx

    def update(self, idx, p_idx, reward):
        if self.inactive:
            return tf.zeros(())
            
        with tf.variable_scope(self.name) as scope:
            with tf.control_dependencies([idx, p_idx, reward]):
                scope.reuse_variables()
                w = tf.get_variable("w")
                #reg_w = tf.exp( (self.gamma * 1e-3/ self.K) / w )
                r_vec = tf.reshape(tf.one_hot(idx, self.K, 
                    tf.exp(self.gamma * reward / (p_idx * self.K)), 1.0), [1,self.K])
                un_w = w * r_vec #* reg_w
                op = tf.assign(w, un_w / tf.reduce_sum(un_w)) 

                #op = tf.assign(w, tf.reshape(tf.one_hot(idx, self.K, 1., 0.), [1,self.K]))
                #op = tf.Print(op, [op], "update")
                with tf.control_dependencies([op]):
                    ret = tf.zeros(())
            return ret
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.identity(self.w))

class RWM(object):
    def __init__(self, name, K, gamma):
        with tf.variable_scope(name) as scope:
            ctx = get_current_tower_context()
            self.inactive = ctx.is_training is not None and not ctx.is_training
            if self.inactive:
                return
            self.name = name
            self.K = K
            self.gamma = gamma
            self.default_w = np.ones((1,self.K),dtype=np.float32) / self.K
            self.w = tf.get_variable("w", [1, self.K], 
                initializer=tf.constant_initializer(self.default_w), trainable=False) 
            for k in range(self.K):
                add_moving_summary(self.w[0][k])
    
    def sample(self):
        if self.inactive:
            return -1, 0.0

        with tf.variable_scope(self.name) as scope:
            scope.reuse_variables()
            w = tf.get_variable("w")
            op = tf.assign(w, w / tf.reduce_sum(w))
            #op = tf.Print(op, [op], "sample")
            with tf.control_dependencies([op]):
                probs = (1.0 - self.gamma) * w + self.gamma * self.default_w
                idx = tf.cast(tf.multinomial(tf.log(probs), 1)[0][0], tf.int32)
                p_idx = probs[0][idx]
                return idx, p_idx

    def update(self, idx, reward):
        if self.inactive:
            return tf.zeros(())
            
        with tf.variable_scope(self.name) as scope:
            with tf.control_dependencies([idx, reward]):
                scope.reuse_variables()
                w = tf.get_variable("w")
                r_vec = tf.reshape(tf.one_hot(idx, self.K, 
                                              tf.exp(self.gamma * reward / self.K), 1.0),
                                   [1, self.K])
                op = tf.assign(w, tf.multiply(w, r_vec))
                #op = tf.Print(op, [op], "update")
                with tf.control_dependencies([op]):
                    ret = tf.zeros(())
            return ret


class RandSelect(object):
    def __init__(self, name, K):
        with tf.variable_scope(name) as scope:
            self.name = name
            self.K = K
            self.probs = tf.constant(np.ones([1,K], dtype=np.float32)/K)
            ctx = get_current_tower_context()
            self.inactive = ctx.is_training is not None and not ctx.is_training

    def sample(self):
        if self.inactive:
            return -1, 0.0
        with tf.variable_scope(self.name) as scope:
            idx = tf.cast(tf.multinomial(tf.log(self.probs), 1)[0][0], np.int32)
            return idx, self.probs[idx]

    def update(self, *args):
        return

class HalfEndHalfExp3(object):
    def __init__(self, name, K, gamma):
        with tf.variable_scope(name) as scope:
            self.K = K
            self.name = name
            self.exp3 = Exp3(name+'_exp3', K, gamma)
            ctx = get_current_tower_context()
            self.inactive = ctx.is_training is not None and not ctx.is_training

    def sample(self):
        if self.inactive:
            return -1, 0.0

        with tf.variable_scope(self.name):
            self.coin = tf.cast(tf.multinomial(tf.log([[0.5,0.5]]), 1)[0][0], tf.int32)
            return tf.cond(tf.equal(self.coin, 0), 
                lambda: (tf.constant(self.K-1, dtype=tf.int32), 
                         tf.constant(1.0,dtype=tf.float32)),
                lambda: self.exp3.sample())

    def update(self, idx, p_idx, reward):
        if self.inactive:
            return tf.zeros(())

        with tf.variable_scope(self.name):
            return tf.cond(tf.equal(self.coin, 0), 
                lambda: tf.zeros(()), 
                lambda: self.exp3.update(idx, p_idx, reward)) 
