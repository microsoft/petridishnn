# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import os
from tensorpack.dataflow import DataFlow
from tensorpack.input_source import TensorInput
from tensorpack.utils import logger
from petridish.data.ptb.ptb_tf import _build_vocab, _file_to_word_ids, ptb_producer


B_DIM = 0
S_DIM = 1
C_DIM = 2

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.shape[0] // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = data.reshape([bsz, -1])
    return data

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, source.shape[S_DIM] - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]
    return data, target

def get_ptb_data(subset, ptb_data_dir):
    if subset in ['val', 'validation']:
        subset = 'valid'
    assert subset in ['train', 'valid', 'test'], subset
    word_to_id = _build_vocab(os.path.join(ptb_data_dir, 'ptb.train.txt'))
    fn = 'ptb.{}.txt'.format(subset)
    data = np.asarray(
        _file_to_word_ids(os.path.join(ptb_data_dir, fn), word_to_id))
    return data, len(word_to_id)

def get_ptb_tensor_input(subset, ptb_data_dir, batch_size, seq_len):
    data, vocab_size = get_ptb_data(subset, ptb_data_dir)
    steps_per_epoch = (data.shape[0] // batch_size  - 1) // seq_len
    data_src = TensorInput(
        lambda: ptb_producer(data, batch_size, seq_len),
        steps_per_epoch)
    return data_src, steps_per_epoch, vocab_size

class PennTreeBankDataFlow(DataFlow):

    def __init__(self, subset, ptb_data_dir, batch_size, seq_len, var_size=False):
        data, self.vocab_size = get_ptb_data(subset, ptb_data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data = batchify(data, batch_size)
        self.var_size = var_size
        if var_size:
            self._reset_l_seq_len()
        else:
            #self._size = (self.data.shape[S_DIM] - 1 + self.seq_len - 1) // self.seq_len
            self._size = (self.data.shape[S_DIM] - 1) // self.seq_len

    def _reset_l_seq_len(self):
        if not self.var_size:
            return
        i = 0
        self.l_seq_len = []
        while i < self.data.shape[S_DIM] - 1 - 1:
            bptt = (
                self.seq_len if np.random.random() < 0.95
                else self.seq_len / 2.)
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            seq_len = min(seq_len, self.seq_len + 20)
            seq_len = min(seq_len, self.data.shape[S_DIM] - 1 - i)
            i += seq_len
            self.l_seq_len.append(seq_len)
        self._size = len(self.l_seq_len)

    def size(self):
        return self._size

    def get_data(self):
        if self.var_size:
            i = 0
            np.random.shuffle(self.l_seq_len)
            for seq_len in self.l_seq_len:
                yield get_batch(self.data, i, seq_len)
                i += seq_len
        else:
            for i in range(self._size):
                yield get_batch(self.data, i * self.seq_len, self.seq_len)

def tensorpack_training_params(args):
    untracked_args = lambda : 0
    args.init_lr_per_sample = 1.0 / 20.
    args.batch_size_per_gpu = 20 # should be 64
    args.optimizer = 'gd'
    args.sgd_moment = 0.0
    args.model_rnn_slowness_reg = None
    args.model_rnn_l2_reg = None
    args.model_rnn_keep_prob_i = 0.6
    args.model_rnn_keep_prob_e = 0.9
    args.model_rnn_keep_prob_h = None
    args.model_rnn_keep_prob_x = None
    args.model_rnn_keep_prob = None
    args.model_rnn_max_len = 35
    args.model_rnn_init_range = 0.05
    args.model_rnn_has_static_len = True
    args.model_rnn_lock_embedding = True
    args.regularize_coef = 'const'
    args.regularize_const = 0.0
    untracked_args.grad_clip = 5
    return args, untracked_args

def petridish_training_params(args):
    untracked_args = lambda : 0
    args.init_lr_per_sample = 0.0001 / 64.
    args.batch_size_per_gpu = 64 # should be 64
    args.optimizer = 'adam'
    args.sgd_moment = 0.0
    args.model_rnn_slowness_reg = None
    args.model_rnn_l2_reg = None
    args.model_rnn_keep_prob_i = 0.6
    args.model_rnn_keep_prob_e = 0.9
    args.model_rnn_keep_prob_h = 0.75
    args.model_rnn_keep_prob_x = 0.5
    args.model_rnn_keep_prob = 0.5
    args.model_rnn_max_len = 35
    args.model_rnn_init_range = 0.05
    args.model_rnn_has_static_len = True
    args.model_rnn_lock_embedding = True
    args.regularize_coef = 'const'
    args.regularize_const = 8e-7
    untracked_args.grad_clip = 0.25
    return args, untracked_args

def darts_training_params(args):
    untracked_args = lambda : 0
    args.init_lr_per_sample = 20. / 64.
    args.batch_size_per_gpu = 64 # should be 64
    args.model_rnn_keep_prob = 0.25
    args.model_rnn_keep_prob_i = 0.8
    args.model_rnn_keep_prob_e = 0.9
    args.model_rnn_keep_prob_h = 0.75
    args.model_rnn_keep_prob_x = 0.25
    args.model_rnn_max_len = 35
    args.model_rnn_slowness_reg = 1e-3
    args.model_rnn_l2_reg = 0.
    args.model_rnn_init_range = 0.04
    args.model_rnn_has_static_len = False
    args.model_rnn_lock_embedding = True
    args.regularize_coef = 'const'
    args.regularize_const = 8e-7
    args.optimizer = 'gd'
    args.sgd_moment = 0.0
    untracked_args.grad_clip = 0.25
    # disable batch norm
    return args, untracked_args

def darts_search_params(args):
    args, untracked_args = darts_training_params(args)
    args.init_lr_per_sample = 20. / 256
    args.batch_size_per_gpu = 16 # should be 256
    args.regularize_const = 5e-7
    args.max_train_model_epoch = 50
    # enable batch norm
    return args, untracked_args

def salesforce_training_params(args):
    untracked_args = lambda : 0
    args.init_lr_per_sample = 30. / 20.
    args.batch_size_per_gpu = 20
    args.model_rnn_keep_prob_i = 1 - 0.4
    args.model_rnn_keep_prob_e = 1 - 0.1
    args.model_rnn_keep_prob_h = 1 - 0.25
    args.model_rnn_keep_prob_x = 1 # need to check
    args.model_rnn_keep_prob = 1 # need to check
    args.model_rnn_init_range = 0.1
    args.model_rnn_max_len = 70
    args.model_rnn_l2_reg = 2.
    args.model_rnn_slowness_reg = 1.
    args.model_rnn_has_static_len = False
    args.model_rnn_lock_embedding = True
    args.regularize_coef = 'const'
    args.regularize_const = 1.2e-6
    args.optimizer = 'gd'
    args.sgd_moment = 0.0
    untracked_args.grad_clip = 0.25
    return args, untracked_args

def training_params_update(args):
    # update training specific parameters
    tt = args.training_type
    if tt == 'salesforce':
        return salesforce_training_params(args)
    elif tt == 'darts_final':
        return darts_training_params(args)
    elif tt == 'tensorpack':
        return tensorpack_training_params(args)
    elif tt == 'petridish':
        return petridish_training_params(args)
    elif tt is None:
        untracked_args = lambda : 0
        return args, untracked_args
    raise ValueError("Unknown Training Type for PTB: {}".format(tt))
