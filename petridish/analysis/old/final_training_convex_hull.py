"""
This script aims to compile final training results, i.e.,
individual training of found models using the best
known training procedure, in order to form the performance
convex hull of the found models.

e.g.,
to plot imagenet final training result:

python3 petridish/analysis/final_training_convex_hull.py \
--keys=951,965 --cust_exps=1176..1183,1185..1193 \
--perf_type=avg --required_epoch=40 --is_pretty \
--labels='Petridish Cell,Petridish Macro' --presenter_func=imagenet
"""
import argparse
import re
import os
import numpy
import copy
import matplotlib.pyplot as plt
from functools import partial
from petridish.analysis.common import (
    img_dir, ann_models_logs, experiment_list_fn,
    for_cust_exps, for_trial_stdouts,
    cust_exps_str_to_list)
from petridish.analysis.old.model_analysis import (
    func_stdout_for_model_perf,
    init_state_for_model_perf,
    merge_state)
from petridish.utils.geometry import _convex_hull_from_points

args = None

def get_final_training_flop_val_err(eidx):
    func_exp_dir_for_model_perf = partial(
        for_trial_stdouts,
        func_stdout=func_stdout_for_model_perf
    )
    state = for_cust_exps(
        [str(eidx)],
        func_exp_dir_for_model_perf,
        init_state_for_model_perf()
    )
    ret_dict = merge_state(state, required_epoch=args.required_epoch)
    #print(ret_dict)
    if not eidx in ret_dict.keys():
        return None, None, None
    fp, avg_ve, min_ve = ret_dict[eidx]
    if avg_ve >= 2:
        avg_ve = None
    if min_ve >= 2:
        min_ve = None
    return fp, avg_ve, min_ve

def is_in_cust_exps(cust_exps, eidx):
    if not cust_exps:
        return True
    return int(eidx) in cust_exps or str(eidx) in cust_exps

def find_matching_key(keys, desc):
    for key in keys:
        if key in desc:
            return key
    return None

def default_presenter(data_points, labels):
    for ki, key in enumerate(data_points.keys()):
        l_x, l_y = data_points[key]
        if labels:
            label = labels[ki]
        else:
            label = key
        plt.plot(l_x, l_y, label=label, marker='P', markersize=7, linewidth=2)
    plt.xlabel('Giga FLOPs', fontsize=fontsize)
    plt.ylabel('Test Error Rate', fontsize=fontsize)

def imagenet_present(data_points, labels):
    # switch from giga flops to million multi-adds
    # switch from error rate between 0 and 1 to percentage.
    for key in data_points.keys():
        l_x, l_y = data_points[key]
        l_x = [x / 2 * 1000 for x in l_x]
        l_y = [y * 100. for y in l_y]
        data_points[key] = [l_x, l_y]

    # plot competitors with also convex hulls
    line_l_name_x_y = [

    ]
    full_data_points = copy.deepcopy(data_points)
    for val in line_l_name_x_y:
        name, xs, ys = val
        full_data_points[name] = [xs, ys]
    default_presenter(full_data_points, labels)

    # plot competiters.
    scatter_l_name_x_y = [
        ('NASNet-A', 564,  26.0),
        ('NASNet-B', 488, 27.2),
        ('AmoebaNet-A', 555, 25.5),
        ('PNAS', 588, 25.8),
        ('DARTS', 595, 26.9),
        ('SNAS', 522, 27.3),
    ]

    for val in scatter_l_name_x_y:
        name, x, y = val
        plt.scatter(x, y, marker='x', label=name)

    plt.grid()
    plt.xlabel('Million Multi-adds', fontsize=fontsize)
    plt.ylabel('Test Error Percentage', fontsize=fontsize)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cust_exps', type=str, default=None,
        help=('a comma separated list of intervals: start..end '
            'These cover the final training experiments that '
            'we are allowed to look at. Default to be all. '
            'bash style intervals, so they are inclusive.')
    )
    parser.add_argument(
        '--keys', type=str, default=None,
        help=('a comma seprated list of strings. e.g., key1,key2,key3 '
            'experiments whose description has key1 as a substring '
            'will be part of the points of the convex hull plot named key1')
    )
    parser.add_argument(
        '--experiment_list_fn', type=str,
        default='./petridish/cust_exps_gen/experiment_list.txt',
        help=('A list of experiment descriptions. none empty line with '
            'pattern r\'  (eidx) : (desc)\' is a description of the experiment id '
            'eidx. Default is to be within the repo under cust_exps_gen (see code)')
    )
    parser.add_argument(
        '--perf_type', type=str,
        default='min', choices=['min', 'avg']
    )
    parser.add_argument(
        '--required_epoch', type=int,
        default=500, help='Use 40 for imagenet, 500 for cifar'
    )
    parser.add_argument(
        '--is_pretty', default=False,
        action='store_true', help='Enable this to iron out points that are not on convex hull'
    )
    parser.add_argument(
        '--presenter_func', default=None,
        help='additional function to call to plot more stuff'
    )
    parser.add_argument(
        '--labels', default=None,
        help=('Label names on the figure for each key in a comma seprated string. '
            'Known bugs: the --labels may not match keys ordering... because this '
            'the key ordering is determined by the python dict, but labels are fixed. '
            'To avoid this, just call the command with --labels first, and add them '
            'when you know the order of the dict from the first run')
    )
    args = parser.parse_args()

    assert os.path.exists(args.experiment_list_fn), \
        'Experiment list file does not exist: {}'.format(
            args.experiment_list_fn)

    # preprocess cust_exps
    args.cust_exps = cust_exps_str_to_list(args.cust_exps)

    # preprocess keys
    args.keys = args.keys.strip().split(',')

    ## check each line of experiemnt description
    ## and accumulate the data points for each key.
    # data_points[key] = [[x1, x2, ...], [y1, y2, ...]]
    data_points = dict()
    with open(args.experiment_list_fn, 'rt') as fin:
        for line in fin:
            if not (line.startswith('  ') and ':' in line):
                continue
            eidx_desc = line.split(':')
            eidx = int(eidx_desc[0].strip())
            desc = eidx_desc[1].strip()
            if not is_in_cust_exps(args.cust_exps, eidx):
                continue
            key = find_matching_key(args.keys, desc)
            if not key:
                continue
            # get the perf for this eidx.
            x, avg_y, min_y = get_final_training_flop_val_err(eidx)
            #print("have results with {} {} {}".format(x, avg_y, min_y))
            if args.perf_type == 'min':
                y = min_y
            elif args.perf_type == 'avg':
                y = avg_y
            if y is None:
                continue
            if not key in data_points:
                data_points[key] = [[], []]
            data_points[key][0].append(x)
            data_points[key][1].append(y)

    print(data_points)

    ## plot the lines for each key.
    ## possible processing:
    ## 1. sort by x (always needed)
    ## 2. remove non convex hull points

    plt.close('all')
    fontsize = 14
    for key in data_points.keys():
        l_x, l_y = data_points[key]
        # sort by x val
        indices = list(range(len(l_x)))
        indices = sorted(indices, key=lambda idx : float(l_x[idx]))
        l_x = [l_x[idx] for idx in indices]
        l_y = [l_y[idx] for idx in indices]

        # TODO remove non-convex hull points
        if args.is_pretty:
            indices, _ = _convex_hull_from_points(l_x, l_y)
            l_x = [l_x[idx] for idx in indices]
            l_y = [l_y[idx] for idx in indices]

        data_points[key] = [l_x, l_y]

    if args.labels:
        args.labels = args.labels.split(',')

    if not args.presenter_func:
        default_presenter(data_points, args.labels)
    elif args.presenter_func == 'imagenet':
        imagenet_present(data_points, args.labels)


    plt.legend(fontsize=fontsize * 2 // 3)
    plt.savefig(
        os.path.join(img_dir, 'final_CH_keys_{}.png'.format('_'.join(args.keys))),
        dpi=plt.gcf().dpi, bbox_inches='tight')
