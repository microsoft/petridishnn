import argparse
import os, sys, re
import numpy as np

from petridish.analysis.model_util import (
    verb_print, model_multi_add, model_nparam, model_errors
)

from petridish.analysis.philly_util import (
    app_dir_log_fns, cust_exp_app_dir, cust_exps_str_to_list,
    app_dir_stdout_fns
)


def compute_cust_exp_average(l_ret, min_epoch, min_nparam):
    eidx_to_perfs = dict()

    for ret in l_ret:
        (
            fn, ma, nparam, ve_val, ve_train, ve_final,
            ve_last_5, ve_last_5_std,
            ve_val_idx, ve_train_idx, ve_final_idx,
        ) = ret
        if not (ve_final_idx and ve_final and ve_final_idx >= min_epoch):
            continue
        reret = re.search(r'petridish_([0-9]*)_', fn)
        if not reret:
            continue
        eidx = reret.group(1)
        if eidx not in eidx_to_perfs:
            eidx_to_perfs[eidx] = []
        eidx_to_perfs[eidx].append([ma, nparam, ve_final, ve_final_idx])

    for eidx in sorted(eidx_to_perfs.keys(), key=lambda _x : int(_x)):
        perfs = eidx_to_perfs[eidx]
        perf_str = None
        if len(perfs) > 0:
            perfs = np.asarray(perfs)
            ma, nparam = np.min(perfs[:,:2], axis=0)
            ve, epoch = np.mean(perfs[:,2:], axis=0)
            ve_std = np.std(perfs[:, 2])
            n_samples = perfs.shape[0]
            if nparam < min_nparam:
                continue
            perf_str = 'ma={:0.2f}M nparam={:0.2f}M ve={:0.2f} ve_std={:0.2f} avg_epoch={} n_samples={}'.format(
                ma * 1e-6, nparam * 1e-6, ve * 100, ve_std * 100, int(epoch), n_samples
            )

        verb_print(
            "eidx={} {}".format(
                eidx, perf_str),
            verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_fn', type=str, default=None,
        help='log file path. Can be a comma separated list of files')
    parser.add_argument(
        '--app_dir', type=str, default=None,
        help='comma separated list of app dir of philly structure'
    )
    parser.add_argument(
        '--cust_exps', type=str, default=None,
        help=('comma separated list of cust_exps ids. can use inclusive'
         'interval with start..end')
    )
    parser.add_argument(
        '--philly_log_root', type=str, default='./petridish_models_logs'
    )
    parser.add_argument(
        '--avg_exp', default=False, action='store_true'
    )
    parser.add_argument(
        '--min_epoch', default=500, type=int,
        help='500 for 1-gpu cifar, 50 for 4-gpu imagenet'
    )
    parser.add_argument(
        '--min_nparam', default=0, type=float,
        help='The minimum of number of parameters for a model to be printed'
    )
    args = parser.parse_args()

    l_fn = set()
    if args.log_fn:
        l_fn.update(args.log_fn.split(','))
    if args.app_dir:
        l_app_dir = args.app_dir.split(',')
        for app_dir in l_app_dir:
            l_fn.update(app_dir_stdout_fns(app_dir))
    if args.cust_exps:
        cust_exps = cust_exps_str_to_list(args.cust_exps)
        for eidx in cust_exps:
            l_app_dir = cust_exp_app_dir(eidx, args.philly_log_root)
            for app_dir in l_app_dir:
                l_fn.update(app_dir_stdout_fns(app_dir))
    l_fn = sorted(list(l_fn))

    results = []
    for fn in l_fn:
        print('-'*80)
        print('fn={}'.format(fn))
        ma, nparam = None, None
        (ve_val, ve_train, ve_final,
            ve_last_5, ve_last_5_std,
            ve_val_idx, ve_train_idx, ve_final_idx) = [None] * 8
        try:
            ma = model_multi_add(fn)
            nparam = model_nparam(fn)
            (ve_val, ve_train, ve_final,
            ve_last_5, ve_last_5_std,
            ve_val_idx, ve_train_idx, ve_final_idx) = model_errors(fn)
        except Exception as e:
            verb_print('Errors!!!!', verbose=True)
        verb_print("ma={} nparam={} ve={} ve_last_5={} \\pm {}".format(
            ma, nparam, ve_final, ve_last_5, ve_last_5_std), verbose=True)
        verb_print("ve_val={} ve_train={}".format(ve_val, ve_train), verbose=True)
        verb_print("ve_val_epoch={} ve_train_epoch={} ve_final_epoch={}".format(
            ve_val_idx, ve_train_idx, ve_final_idx), verbose=True)
        results.append((
            fn, ma, nparam, ve_val, ve_train, ve_final,
            ve_last_5, ve_last_5_std,
            ve_val_idx, ve_train_idx, ve_final_idx,
        ))

    if args.avg_exp:
        compute_cust_exp_average(results, args.min_epoch, args.min_nparam)
