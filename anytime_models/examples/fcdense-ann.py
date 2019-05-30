import numpy as np
import argparse
import os, sys, datetime, shutil, subprocess

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils import utils
from tensorpack.utils import fs
from tensorpack.callbacks import JSONWriter, ScalarPrinter

import anytime_models.models.anytime_network as anytime_network
from anytime_models.models.anytime_network import \
    AnytimeLogDenseNetV1, AnytimeLogLogDenseNet
from anytime_models.models.anytime_fcn import \
    AnytimeFCNCoarseToFine, AnytimeFCDenseNet, AnytimeFCDenseNetV2, \
    parser_add_fcdense_arguments
import get_augmented_data
import ann_app_utils


"""
"""
INPUT_SIZE=None
args=None
lr_schedule=None
max_epoch=None
get_data=None
side = 224

def get_camvid_data(which_set, shuffle=True, slide_all=False):
    isTrain = (which_set == 'train' or which_set == 'trainval') and shuffle

    pixel_z_normalize = True
    ds = dataset.Camvid(which_set, shuffle=shuffle,
        pixel_z_normalize=pixel_z_normalize,
        is_label_one_hot=args.is_label_one_hot,
        slide_all=slide_all,
        slide_window_size=side,
        void_overlap=not isTrain)
    data_orig_x, data_orig_y = dataset.Camvid.data_shape[0:2]

    x_augmentors = []
    xy_augmentors = []
    if args.operation == 'train':
        if isTrain:
            x_augmentors = [
                #imgaug.GaussianBlur(2)
            ]
            xy_augmentors = [
                #imgaug.RotationAndCropValid(7),
                #imgaug.RandomResize((0.8, 1.5), (0.8, 1.5), aspect_ratio_thres=0.0),
                imgaug.RandomCrop((side, side)),
                imgaug.Flip(horiz=True),
            ]
        else:
            xy_augmentors = [
                imgaug.RandomCrop((side, side)),
            ]
    elif args.operation == 'finetune':
        if isTrain:
            xy_augmentors = [
                imgaug.Flip(horiz=True),
            ]
        else:
            xy_augmentors = [
            ]
    elif args.operation == 'evaluate':
        xy_augmentors = [
        ]

    if len(x_augmentors) > 0:
        ds = AugmentImageComponent(ds, x_augmentors, copy=True)
    ds = AugmentImageComponents(ds, xy_augmentors, copy=False)
    ds = BatchData(ds, args.batch_size // args.nr_gpu, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds

def get_pascal_voc_data(subset, do_multiprocess=True):
    return get_augmented_data.get_pascal_voc_augmented_data(subset, args, do_multiprocess)

def label_image_to_rgb(label_img, cmap):
    if len(label_img.shape) > 2:
        label_img = np.argmax(label_img, axis=-1)
    H, W = (label_img.shape[0], label_img.shape[1])
    return np.asarray([cmap[y] for y in label_img.reshape([-1])], dtype=np.uint8).reshape([H,W,3])

def evaluate(subset, get_data, model_cls, meta_info):
    if args.display_period > 0 and logger.LOG_DIR is not None:
        # we import here since this is only used for presentation, which never happens on servers.
        # server may not have these packages.
        import matplotlib.pyplot as plt
        import skimage.transform
        imresize = skimage.transform.resize
        if args.do_video_multi_add:
            fn = default_multi_add_fn()
            pred_layers, pred_costs = grep_flops(fn)

    args.batch_size = 1
    cmap = meta_info._cmap
    mean = meta_info.mean
    std = meta_info.std

    ds = get_data(subset, args, False)
    model = model_cls(args)

    l_outputs = []
    n_preds = np.sum(model.weights > 0)
    for i, weight in enumerate(model.weights):
        if weight > 0:
            l_outputs.extend(\
                ['layer{:03d}.0.pred/confusion_matrix/SparseTensorDenseAdd:0'.format(i),
                 'layer{:03d}.0.pred/pred_prob_img:0'.format(i)])

    pred_config = PredictConfig(
        model=model,
        session_init=SaverRestore(args.load),
        input_names=['input', 'label'],
        output_names=['input', 'label'] + l_outputs
    )
    pred = SimpleDatasetPredictor(pred_config, ds)

    l_total_confusion = [0] * n_preds
    for i, output in enumerate(pred.get_result()):
        img = np.asarray((output[0][0] * std + mean)*255, dtype=np.uint8)
        label = output[1][0]
        if len(label.shape) == 3:
            mask = label.sum(axis=-1) < args.eval_threshold
        else:
            mask = label < args.num_classes
        label_img = label_image_to_rgb(label, cmap)

        confs = output[2:][::2]
        preds = output[2:][1::2]

        for predi, perf in enumerate(zip(confs, preds)):
            conf, pred = perf
            l_total_confusion[predi] += conf

        if args.display_period > 0 and i % args.display_period == 0 \
                and logger.LOG_DIR is not None:
            save_dir = logger.LOG_DIR

            #select_indices = [ int(np.round(n_preds * fraci / 4.0)) - 1 \
            #                for fraci in range(1,5) ]
            select_indices = list(range(n_preds))
            preds_to_display = [preds[idx] for idx in select_indices]
            plt.close('all')
            if not args.do_video:
                n_fig = 2 + len(select_indices)
                fig, axarr = plt.subplots(
                    1, n_fig, figsize=(1 + 5.5 * n_fig , 5))
                axarr[0].imshow(img)
                axarr[1].imshow(label_img)
                for predi, pred in enumerate(preds_to_display):
                    pred_img = pred[0].argmax(axis=-1)
                    pred_img = label_image_to_rgb(pred_img, cmap)
                    pred_img = imresize(pred_img, (img.shape[0], img.shape[1]), order=0)
                    axarr[2+predi].imshow(pred_img)

                plt.savefig(os.path.join(logger.LOG_DIR, 'img_{}.png'.format(i)),
                    dpi=fig.dpi, bbox_inches='tight')

            else:
                dn = os.path.join(logger.LOG_DIR, 'img_{}'.format(i))
                if not os.path.exists(dn):
                    os.makedirs(dn)

                plt.imshow(img)
                plt.savefig(
                    os.path.join(dn, 'input.png'),
                    dpi=plt.gcf().dpi,
                    bbox_inches='tight'
                )
                plt.close('all')

                plt.imshow(label_img)
                plt.savefig(
                    os.path.join(dn, 'label.png'),
                    dpi=plt.gcf().dpi,
                    bbox_inches='tight'
                )
                plt.close('all')

                for predi, pred in enumerate(preds_to_display):
                    pred_img = pred[0].argmax(axis=-1)
                    pred_img = label_image_to_rgb(pred_img, cmap)
                    pred_img = imresize(pred_img, (img.shape[0], img.shape[1]), order=0)
                    plt.imshow(pred_img)
                    if args.do_video_multi_add and pred_costs:
                        plt.xlabel(
                            'Number of Multi-adds : {:0.2f}e9'.format(pred_costs[predi] * 1e-9),
                            fontsize=14
                        )
                    plt.savefig(
                        os.path.join(dn, 'pred_{}.png'.format(predi)),
                        dpi=plt.gcf().dpi,
                        bbox_inches='tight'
                    )
                    plt.close('all')


    #endfor each sample
    l_ret = []
    for i, total_confusion in enumerate(l_total_confusion):
        ret = dict()
        ret['subset'] = subset
        ret['confmat'] = total_confusion
        I = np.diag(total_confusion)
        n_true = np.sum(total_confusion, axis=1)
        n_pred = np.sum(total_confusion, axis=0)
        U = n_true + n_pred - I
        U += U==0
        IoUs = np.float32(I) / U
        mIoU = np.mean(IoUs)
        ret['IoUs'] = IoUs
        ret['mIoU'] = mIoU
        logger.info("ret info: {}".format(ret))

    if logger.LOG_DIR:
        npzfn = os.path.join(logger.LOG_DIR, 'evaluation.npz')
        np.savez(npzfn, evaluation_ret=ret)

def get_config(ds_trian, ds_val, model_cls):
    # prepare dataset
    steps_per_epoch = ds_train.size() // args.nr_gpu
    starting_epoch = ann_app_utils.grep_starting_epoch(args.load, steps_per_epoch)
    logger.info("The starting epoch is {}".format(starting_epoch))
    args.init_lr = ann_app_utils.grep_init_lr(starting_epoch, lr_schedule)
    logger.info("The starting learning rate is {}".format(args.init_lr))
    model=model_cls(args)
    classification_cbs = model.compute_classification_callbacks()
    loss_select_cbs = model.compute_loss_select_callbacks()

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(checkpoint_dir=args.model_dir, max_to_keep=2, keep_checkpoint_every_n_hours=12),
            InferenceRunner(ds_val,
                            [ScalarStats('cost')] + classification_cbs),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate')
        ] + loss_select_cbs,
        model=model,
        monitors=[JSONWriter(), ScalarPrinter()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
        starting_epoch=starting_epoch,
    )


def grep_flops(fn):
    """
    Args:
    fn (str) : path to the file that contains the multi-add info.
        To create the file, execute cmd
        ``` cat log.log | grep "multi-add" > $fn ```
    """
    # first path to figure out the critical layers
    pred_layers = []
    pred_costs = []
    if not os.path.exists(fn):
        return None, None
    with open(fn, 'rt') as fin:
        for li, line in enumerate(fin):
            line = line.strip()
            if 'pred/linear' in line:
                start = line.find('layer')
                end = line.find('.', start)
                idx = line[start:end]
                pred_layers.append(idx)
                cost = float(line[line.rfind(' ')+1:])
                pred_costs.append(cost)
                if len(pred_layers) == 1:
                    line_end_idx = li

    with open(fn, 'rt') as fin:
        pred_i = 0
        cost = 0
        for li, line in enumerate(fin):
            if li >= line_end_idx:
                break
            line = line.strip()
            line_cost = float(line[line.rfind(' ')+1:])
            cost += line_cost
            if pred_layers[pred_i] in line:
                pred_costs[pred_i] += cost
                pred_i += 1

    for layer, cost in zip(pred_layers, pred_costs):
        print('{}\t{}'.format(layer, cost))
    return pred_layers, pred_costs

def default_multi_add_fn():
    fn = os.path.join(logger.LOG_DIR, 'multi_add.txt')
    return fn

def form_video(
        fn_pattern='{}.png',
        video_fn=None,
        switch_times=None,
        work_dir='video_temp',
        total_frames=200):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    n_image = len(switch_times)
    first_switch = switch_times[0]
    for i in range(len(switch_times)):
        switch_times[i] -= first_switch
    last_switch = switch_times[-1] * 1.6
    curr_img = 0
    for imgi in range(n_image):
        switch_time = switch_times[imgi]
        next_switch_time = switch_times[imgi + 1] if imgi < n_image - 1 else last_switch
        switch_img = int(0.5 + switch_time / last_switch * total_frames)
        next_switch = int(0.5 + next_switch_time / last_switch * total_frames)
        src_img_fn = fn_pattern.format(imgi)
        for framei in range(switch_img, next_switch):
            frame_fn = os.path.join(work_dir, '{:03d}.png'.format(framei))
            logger.info("src : {} \n frame : {}".format(src_img_fn, frame_fn))
            os.symlink(src_img_fn, frame_fn)

    frames_to_video(
        frame_dir=work_dir,
        video_fname=video_fn,
        video_fps=5,
        frame_format='%03d.png'
    )

    for framei in range(total_frames):
        frame_fn = os.path.join(work_dir, '{:03d}.png'.format(framei))
        if os.path.exists(frame_fn):
            os.remove(frame_fn)


def frames_to_video(frame_dir, video_fname = None, video_fps = 20, frame_format='frame%03d.png'):
    frame_dir = os.path.expanduser(frame_dir)
    if not os.path.isdir(frame_dir):
        raise IOError('Cannot find directory: {}'.format(frame_dir))
    if frame_dir[-1] != os.path.sep:
        frame_dir += os.path.sep
    if video_fname is None:
        parent_dir, base_dir = os.path.split(os.path.dirname(frame_dir))
        video_fname = os.path.join(parent_dir, base_dir + '.mp4')
        print 'Writing video to: {}'.format(video_fname)
    cmd = 'ffmpeg -framerate {0:d} -i {1}/{2} -q:scale 0 -c:v libx264 -vf fps=24 -pix_fmt yuv420p {3}'.format(
        video_fps, frame_dir,
        frame_format, video_fname)
    subprocess.check_call([cmd],shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset choice
    parser.add_argument('--ds_name', help='name of dataset',
                        type=str,
                        choices=['camvid', 'pascal'])
    # other common args
    parser.add_argument('--batch_size', help='Batch size for train/testing',
                        type=int, default=3)
    parser.add_argument('--log_dir', help='log_dir position',
                        type=str, default=None)
    parser.add_argument('--data_dir', help='data_dir position',
                        type=str, default=None)
    parser.add_argument('--model_dir', help='model_dir position',
                        type=str, default=None)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--nr_gpu', help='Number of GPU to use', type=int, default=1)
    parser.add_argument('--is_test', help='Whehter use train-val or train-test',
                        default=False, action='store_true')
    parser.add_argument('--is_philly', help='Whether the script is running in a phily script',
                        default=False, action='store_true')
    parser.add_argument('--operation', help='Current operation',
                        default='train',
                        choices=['train', 'finetune', 'evaluate', 'grep_cost', 'form_video'])
    parser.add_argument('--display_period', help='Display at eval every # of image; 0 means no display',
                        default=0, type=int)
    parser.add_argument('--do_video', help='whether to join the images of evaluation into video',
                        default=False, action='store_true')
    parser.add_argument('--do_video_multi_add', help='Whether images should include flops info',
                        default=False, action='store_true')
    parser_add_fcdense_arguments(parser)
    args = parser.parse_args()

    model_cls = None
    if args.densenet_version == 'atv2':
        model_cls = AnytimeFCDenseNetV2
    elif args.densenet_version == 'atv1':
        model_cls = AnytimeFCDenseNet(AnytimeLogDenseNetV1)
    elif args.densenet_version == 'loglog':
        model_cls = AnytimeFCDenseNet(AnytimeLogLogDenseNet)
    elif args.densenet_version == 'c2f':
        model_cls = AnytimeFCNCoarseToFine
        side = 360
    elif args.densenet_version == 'dense':
        model_cls = FCDensenet

    logger.set_log_root(log_root=args.log_dir)
    logger.auto_set_dir(action='k')
    fs.set_dataset_path(args.data_dir)

    if args.operation == 'grep_cost':
        fn = default_multi_add_fn()
        grep_flops(fn)
        sys.exit(0)

    if args.operation == 'form_video':
        fn = default_multi_add_fn()
        _, pred_costs = grep_flops(fn)

        fn_pattern = os.path.join(logger.LOG_DIR, 'img_0' , 'pred_{}.png')
        work_dir = os.path.join(logger.LOG_DIR, 'img_0')
        video_fn = os.path.join(logger.LOG_DIR, 'img_0', 'img_0_vfr.mp4')
        if os.path.exists(video_fn):
            os.remove(video_fn)
        form_video(
            fn_pattern,
            video_fn,
            pred_costs,
            work_dir
        )
        video_fn = os.path.join(logger.LOG_DIR, 'img_0', 'img_0_cfr.mp4')
        if os.path.exists(video_fn):
            os.remove(video_fn)
        form_video(
            fn_pattern,
            video_fn,
            np.arange(len(pred_costs)),
            work_dir
        )
        sys.exit(0)


    ##
    # Store a philly_operation.txt in root log dir
    # every run will check it if the script is run on philly
    # philly_operation.txt  should contain the same step that is current running
    # if it does not exit, the default is written there (train)
    if args.is_philly:
        philly_operation_fn = os.path.join(args.log_dir, 'philly_operation.txt')
        if not os.path.exists(philly_operation_fn):
            with open(philly_operation_fn, 'wt') as fout:
                fout.write(args.operation)
        else:
            with open(philly_operation_fn, 'rt') as fin:
                philly_operation = fin.read().strip()
            if philly_operation != args.operation:
                sys.exit()

    ## Set dataset-network specific assert/info
    if args.ds_name == 'camvid':
        args.num_classes = dataset.Camvid.non_void_nclasses
        # the last weight is for void
        args.class_weight = dataset.Camvid.class_weight[:-1]
        args.optimizer = 'rmsprop'
        INPUT_SIZE = None
        get_data = get_camvid_data


        if args.operation == 'evaluate':
            args.batch_size = 1
            args.nr_gpu = 1
            args.input_height = dataset.Camvid.data_shape[0]
            args.input_width = dataset.Camvid.data_shape[1]
            subset = 'test' if args.is_test else 'val'
            evaluate(subset, get_data, model_cls, dataset.Camvid)
            sys.exit()

        max_train_epoch = 698
        if args.operation == 'train':
            args.input_height = side
            args.input_width = side
            max_epoch = max_train_epoch
            lr = args.init_lr
            lr_schedule = []
            for i in range(max_epoch):
                lr *= 0.995
                lr_schedule.append((i+1, lr))

        elif args.operation == 'finetune':
            args.input_height = dataset.Camvid.data_shape[0]
            args.input_width = dataset.Camvid.data_shape[1]
            batch_per_gpu = args.batch_size // args.nr_gpu
            init_epoch = max_train_epoch // batch_per_gpu
            lr_schedule = []
            init_lr = 5e-5
            lr = init_lr
            max_epoch=500 + init_epoch
            lr_schedule.append((1, init_lr))
            for i in range(init_epoch, max_epoch):
                lr_schedule.append((i, lr))
                lr *= 0.995

            # Finetune has 1 sample per gpu for each batch
            args.batch_size = args.nr_gpu
            args.init_lr = init_lr

        if not args.is_test:
            ds_train = get_data('train') #trainval
            ds_val = get_data('val') #test
        else:
            ds_train = get_data('train')
            ds_val = get_data('test')


    elif args.ds_name == 'pascal':
        args.num_classes = 21
        args.class_weight = dataset.PascalVOC.class_weight[:-1]
        args.optimizer = 'rmsprop'
        INPUT_SIZE = None
        get_data = get_pascal_voc_data

        if args.operation == 'evaluate':
            subset = 'val'
            evaluate(subset, get_data, model_cls, PascalVOC)
            sys.exit()

        ds_train = get_data('train_extra')
        ds_val = get_data('val')

        max_epoch = 100
        lr = args.init_lr
        lr_schedule = []
        for i in range(max_epoch):
            #lr *= args.init_lr * (1.0 - i / np.float32(max_epoch))**0.9
            lr *= 0.995
            lr_schedule.append((i+1, lr))

    logger.info("Arguments: {}".format(args))
    logger.info("TF version: {}".format(tf.__version__))
    config = get_config(ds_train, ds_val, model_cls)
    if args.load and os.path.exists(args.load):
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(args.nr_gpu))

    ## Since the current operation is done, we write the next operation to the operation.txt
    if args.is_philly:
        def get_next_operation(op):
            if op == 'train':
                return 'finetune'
            if op == 'finetune':
                return 'evaluate'
            return 'done'
        next_operation = get_next_operation(args.operation)
        with open(philly_operation_fn, 'wt') as fout:
            fout.write(next_operation)
