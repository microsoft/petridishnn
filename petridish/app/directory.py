# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import re

from petridish.philly.container import is_philly
from petridish.app.multi_proc import has_stopped

"""
Dir structures
"""
def _updir(d, n=1):
    for _ in range(n):
        d = os.path.dirname(d)
    return d


"""
    Philly specific dir structures regarding multiple trials of the same experiment
"""
def previous_trial_log_root(log_root):
    if not is_philly():
        return None
    # e.g., xx/application_xx-xx/logs/2/petridish_main
    log_root = os.path.normpath(log_root)
    triali = int(os.path.basename(_updir(log_root, 1)))
    if triali == 1:
        return None
    return os.path.join(_updir(log_root, 2), str(triali - 1), os.path.basename(log_root))


def previous_trial_model_root(model_root):
    if not is_philly():
        return None
    # e.g., xxx/application_xx-xx/models
    return os.path.normpath(model_root)
    #model_root = os.path.normpath(model_root)
    #triali = int(os.path.basename(model_root))
    #if triali == 1:
    #    return None
    #return os.path.join(_updir(model_root, 1), str(triali - 1))


"""
    Helper functions to create names for communication over file-system.
    Direct connections are not available.
"""

def _auto_script_fn(i, prefix=None):
    if prefix is not None:
        return '{}_{}.sh'.format(prefix, i)
    return '{}.sh'.format(i)

def _auto_script_dir(log_dir, is_critic, is_log_dir_root=False):
    n_updir = 1 + int(bool(is_critic)) - int(bool(is_log_dir_root)) #+ 2 * is_philly()
    return os.path.join(_updir(log_dir, n_updir), 'auto_scripts')

def _all_mi(dir_root):
    all_mi = []
    for dn in os.listdir(dir_root):
        try:
            mi = int(os.path.basename(dn.strip()))
            all_mi.append(mi)
        except:
            continue
    return all_mi


def _dn_to_mi(dn):
    try:
        mi = int(os.path.basename(os.path.normpath(dn)))
        return mi
    except:
        return None

def _mi_to_dn(dir_root, model_iter):
    return os.path.join(dir_root, str(model_iter))

def _dn_to_ci(dn):
    try:
        ci = int(os.path.basename(os.path.normpath(dn)))
        return ci
    except:
        return None

def _ci_to_dn(dir_root, critic_iter, queue_name):
    if critic_iter is None:
        return os.path.join(dir_root, queue_name)
    return os.path.join(dir_root, queue_name, str(critic_iter))


def _all_critic_dn(dir_root, queue_name):
    return glob.glob(os.path.join(dir_root, queue_name, '*'))

def _latest_ci(log_dir_root, model_dir_root, queue_name):
    l_dns = _all_critic_dn(log_dir_root, queue_name)
    max_ci = None
    for dn in l_dns:
        dn = os.path.normpath(dn.strip())
        try:
            # make sure the dirname is an int so it is actually a dir for critic
            ci = int(os.path.basename(dn))
        except:
            continue

        if not has_stopped(dn):
            # make sure model is mark finished.
            continue

        if not os.path.exists(_ci_to_dn(model_dir_root, ci, queue_name)):
            # make sure model exists
            continue

        if max_ci is None or max_ci < ci:
            max_ci = ci
    return max_ci


def _mi_info_save_fn(log_dir_root):
    return os.path.join(log_dir_root, 'mi_info.npz')