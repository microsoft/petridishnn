import numpy as np
import argparse
import os, sys, datetime

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils
import multiprocessing

INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None

DATA_FORMAT='NCHW'
CHANNEL_DIM = 1

train_range=(0,4500)
test_range = (4500,5000)

def get_ann_policy_data(subset, options, do_multiprocess=True):
    isTrain = subset == 'train' and do_multiprocess
    if subset == 'train':
        select_range = train_range
    else:
        select_range = test_range
    D1 = NPZData('./data/ann_policy/full_cifar100.npz', ['X','Xfeat', 'Y'], select_range)
    D1 = SelectComponent(D1, idxs=[0,1])
    if options.is_reg:
        D2 = NPZData('./data/ann_policy/reg_targets_cifar100.npz', 
                    ['reg_targets'], select_range)
        cls_mean = np.array([ 0.6438303 ,  0.63556123,  0.60529894,  0.60928351,  0.50842327,
                    0.50540853,  0.30445826,  0.30446208, -0.11820364, -0.96956009], dtype=np.float32)
        cls_std = np.array([ 0.46755612,  0.46016496,  0.44967544,  0.43719143,  0.44477478,
                    0.43564209,  0.43981788,  0.42841294,  0.42598745,  0.42474499], dtype=np.float32)

        D2 = MapDataComponent(D2, lambda x: (x - cls_mean)/cls_std)
        

    else:
        D2 = NPZData('./data/ann_policy/targets_cifar100.npz', 
                    ['targets'], select_range)

        cls_cnt = np.array([3748,  548,  241,  167,   96,   59,   52,   52,   37,    0], dtype=np.float32)
        cls_weight = np.median(cls_cnt) / (cls_cnt + (cls_cnt == 0))
        args.class_weight = cls_weight

    ds = JoinData([D1, D2]) 
    if isTrain:
        ds = LocallyShuffleData(ds, 5000)
    if do_multiprocess:
        ds = PrefetchDataZMQ(ds, min(24, multiprocessing.cpu_count()))
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds


class AnytimePolicy(ModelDesc):
    def __init__(self, args):
        self.options = args
        self.num_classes = args.num_classes


    def inputs(self):
        if self.options.is_reg:
            label_desc = InputDesc(tf.float32, [None, self.num_classes], 'label')
        else:
            label_desc = InputDesc(tf.int32, [None], 'label')
        return [InputDesc(tf.float32, [None, 32,32, 3], 'input'), 
                InputDesc(tf.float32, [None, 640, 8, 8], 'nn_feat'), 
                label_desc]

        
    def build_graph(self, *inputs):
        images, feats, labels = inputs     
        if DATA_FORMAT == 'NCHW':
            images = tf.transpose(images, [0,3,1,2])
        with argscope([Conv2D, AvgPooling, MaxPooling, BatchNorm, GlobalAvgPooling],
                    data_format=DATA_FORMAT), \
            argscope([Conv2D, Deconv2D], nl=tf.identity, use_bias=True):

            l1 = Conv2D('conv_img1', images, 32, 3, stride=2, nl=BNReLU)
            l1 = Conv2D('conv_img2', l1, 32, 3, stride=2, nl=BNReLU)

            l2 = Conv2D('conv_feat1', feats, 128, 3, nl=BNReLU)
            l2 = Conv2D('conv_feat2', feats, 64, 3, nl=BNReLU)
            l = tf.concat([l1,l2], CHANNEL_DIM)
        
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, self.num_classes, nl=tf.identity)


            if self.options.is_reg:
                preds = tf.identity(logits, name='output')
                wrong = tf.not_equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1))
                wrong = tf.cast(wrong, dtype=tf.float32, name='incorrect_vector')
                add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
                cost = tf.square(labels - logits)

            else:
                def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
                    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)),
                        tf.float32, name=name)
                wrong = prediction_incorrect(logits, labels)
                add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
                prob = tf.nn.softmax(logits, name='output')
                #cost = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                #    logits=logits, labels=labels)
                max_logits = tf.reduce_max(logits, axis=1, keep_dims=True)
                _logits = logits - max_logits
                normalizers = tf.reduce_sum(tf.exp(_logits), axis=1, keep_dims=True)
                _logits = _logits - tf.log(normalizers)
                labels_one_hot = tf.one_hot(labels, self.num_classes, dtype=tf.float32)
                cross_entropy = -tf.reduce_sum(\
                    tf.multiply(labels_one_hot * _logits, self.options.class_weight), axis=1)
                cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
                add_moving_summary(cross_entropy)
                cost = cross_entropy

            self.cost = tf.reduce_mean(cost, name='cost')
            add_moving_summary(self.cost)
            return self.cost


    def optimizer(self):
        lr = tf.get_variable('learning_rate', 0.01, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.AdamOptimizer(lr)
        return opt

def get_config(args):
    # prepare dataset
    ds_train = get_ann_policy_data('train', args)
    ds_val = get_ann_policy_data('test', args, do_multiprocess=False)
    steps_per_epoch = ds_train.size() // args.nr_gpu

    init_lr = 0.01
    lr_schedule = [ (1,init_lr), (100, init_lr * 0.1), (200, init_lr * 0.01) ]
    max_epoch = 300

    model= AnytimePolicy(args)

    class PrintOutput(Inferencer):
        def __init__(self):
            pass

        def _get_output_tensors(self):
            return ['output', 'label', 'incorrect_vector']

        def _datapoint(self, vec_tensor):
            output, label, wrong = vec_tensor
            print output.shape
            #print wrong
            if args.is_reg:
                # regression on loss objective
                print np.argmax(output, axis=1)
                print np.argmax(label, axis=1)
            else:
                # logits/prob 
                print np.argmax(output, axis=1)
                print label
            #    print output[0]


    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            InferenceRunner(ds_val,
                            [ScalarStats('cost'), 
                             ClassificationError(),
                             PrintOutput()]),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ],
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # other common args
    parser.add_argument('--batch_size', help='Batch size for train/testing', 
                        type=int, default=128)
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--do_validation', help='Whether use validation set. Default not',
                        type=bool, default=False)
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)

    parser.add_argument('--is_reg', help='whether use regression', default=False,
                        action='store_true')
    args = parser.parse_args()
    args.num_classes = 10
    
    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir()
    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))

    config = get_config(args)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    config.nr_tower = args.nr_gpu
    SyncMultiGPUTrainer(config).train()
