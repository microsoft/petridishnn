
import re
import sys
import os
import argparse
import numpy as np
import subprocess

# Grep deliminator for new runs in the same log (only happens on certain server preempt)
grep_new_run = r"""grep "STARTING NEW MASTER COORDINATOR RUN" {fn}"""

# commands used for multi-add
grep_ma_cmd = r"""grep "multi-add" {fn} | grep "[0-9\.]*$" -o | awk '{{s+=$1}} END {{print s}}'"""
grep_aux_preprocess_ma_cmd = r"""grep "aux_preprocess" {fn} | grep "multi-add" | grep "[0-9\.]*$" -o | awk '{{s+=$1}} END {{print s}}'"""
grep_linear_ma_cmd = r"""grep "multi-add" {fn} | grep "linear" | grep "[0-9\.]*$" -o | awk '{{s+=$1}} END {{print s}}'"""
grep_last_linear_ma_cmd = r"""grep "multi-add" {fn} | grep "linear" | grep "[0-9\.]*$" -o | tail -n 1 | awk '{{s+=$1}} END {{print s}}'"""

# commands used for nparam
grep_nparam_start_end = r"""grep -n '\#elements\|Number of train' {fn} | grep -o "^[0-9]*" """
grep_nparam_start_end_old = r"""grep -n 'shape *dim\|Total \#vars' {fn} | grep -o "^[0-9]*" """
grep_nparam_cmd = r"""sed -n {start},{end}p {fn} | grep "[0-9]*$" -o| awk '{{s+=$1}} END {{print s}}'"""
grep_aux_preprocess_nparam_cmd = \
    r"""sed -n {start},{end}p {fn} | grep "aux_preprocess" | grep "[0-9]*$" -o| awk '{{s+=$1}} END {{print s}}'"""
grep_linear_nparam_cmd = \
    r"""sed -n {start},{end}p {fn} | grep "linear" | grep "[0-9]*$" -o| awk '{{s+=$1}} END {{print s}}'"""
grep_last_linear_nparam_cmd = \
    r"""sed -n {start},{end}p {fn} | grep "linear" | grep "[0-9]*$" -o| tail -n 2 | awk '{{s+=$1}} END {{print s}}'"""

## commands for val/train error (evaluated per epoch)
grep_last_layer_name_cmd = \
    r"""grep "multi-add" {fn} | grep "linear" | tail -n 1 | grep "layer[0-9]*" -o"""
grep_val_err_cmd = \
    r"""grep "val_err:" {fn} | grep {name} | grep "[0-9\.]*$" -o"""
grep_train_err_cmd = \
    r"""grep "train_error:" {fn} | grep {name} | grep "[0-9\.]*$" -o"""
grep_epoch_cmd = \
    r"""grep "Start Epoch" {fn} | grep "[0-9]* \.\.\.$" -o | grep "[0-9]*" -o"""

def verb_print(ss, verbose):
    if verbose:
        print(ss)

def get_cmd_output_as_lines(cmd):
    #print("cmd is: \n" + cmd +'\n')
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    lines = [line.decode('utf-8') for line in lines]
    return lines

def model_multi_add(fn):
    """
    Stand-alone function to query for the model multi-add
    """
    l_val = []
    for cmd in [
            grep_ma_cmd,
            grep_aux_preprocess_ma_cmd,
            grep_linear_ma_cmd,
            grep_last_linear_ma_cmd]:
        lines = get_cmd_output_as_lines(cmd.format(fn=fn))
        val_str = lines[0].strip()
        if len(val_str) == 0:
            val = 0.0
        else:
            val = float(val_str)
        l_val.append(val)
    ma = l_val[0] - l_val[1] - (l_val[2] - l_val[3])
    #figure out how many copies we made...
    cmd = grep_new_run.format(fn=fn)
    lines = get_cmd_output_as_lines(cmd)
    n_runs = len(lines) + 1
    return ma / n_runs

def model_nparam(fn):
    """
    Stand-alone function to query for the model number of params
    """
    cmd = grep_nparam_start_end.format(fn=fn)
    lines = get_cmd_output_as_lines(cmd)
    try:
        start, end = [int(line.strip()) for line in lines][-2:]
    except:
        # backward compatible
        cmd = grep_nparam_start_end_old.format(fn=fn)
        lines = get_cmd_output_as_lines(cmd)
        start, end = [int(line.strip()) for line in lines][-2:]

    l_val = []
    for cmd in [
            grep_nparam_cmd,
            grep_aux_preprocess_nparam_cmd,
            grep_linear_nparam_cmd,
            grep_last_linear_nparam_cmd]:
        lines = get_cmd_output_as_lines(cmd.format(
            start=start, end=end, fn=fn))
        val_str = lines[0].strip()
        if len(val_str) == 0:
            val = 0.0
        else:
            val = float(val_str)
        l_val.append(val)
    return l_val[0] - l_val[1] - (l_val[2] - l_val[3])

def model_errors(fn):
    """
    stand-alone function to query model error rates.
    """
    cmd = grep_last_layer_name_cmd.format(fn=fn)
    lines = get_cmd_output_as_lines(cmd)
    name = lines[0].strip()

    cmd = grep_epoch_cmd.format(fn=fn)
    lines = get_cmd_output_as_lines(cmd)
    epochs = [int(line.strip()) for line in lines]

    cmd = grep_val_err_cmd.format(fn=fn, name=name)
    lines = get_cmd_output_as_lines(cmd)
    val_err = [float(line.strip()) for line in lines]

    cmd = grep_train_err_cmd.format(fn=fn, name=name)
    lines = get_cmd_output_as_lines(cmd)
    train_err = [float(line.strip()) for line in lines]

    ve_val_idx = np.argmin(val_err)
    ve_val = val_err[ve_val_idx]
    ve_train_idx = np.argmin(train_err)
    ve_train = val_err[ve_train_idx]
    ve_final_idx = len(val_err) - 1
    ve_final = val_err[-1]
    ve_last_5 = np.mean(val_err[-5:])
    ve_last_5_std = np.std(val_err[-5:])
    return (ve_val, ve_train, ve_final, ve_last_5, ve_last_5_std,
        epochs[ve_val_idx], epochs[ve_train_idx], epochs[ve_final_idx])
