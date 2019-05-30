"""
This script is gathers information about a model training run using
the log file of the run (or the saved stdout of the run).

The information are gathered with `grep` commands on key strings.
We assume that the log file contains one single run.

A typical usage example:

python3 petridish/analysis/model.py \
    --log_fn=<model training log.log>

The three types of val error, ve_val, ve_train and ve_final, represent
the best val error, the val error when training error is the lowest,
and the final val error.
"""
import argparse
import os, sys, re

from petridish.analysis.model_util import (
    verb_print, model_multi_add, model_nparam, model_errors
)

from petridish.analysis.philly_util import (
    app_dir_log_fns, cust_exp_app_dir, cust_exps_str_to_list
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_fn', type=str, default=None,
        help='log file path. Can be a comma separated list of files')
    args = parser.parse_args()

    l_fn = args.log_fn.split(',')

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
