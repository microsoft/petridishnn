"""
This util handles special file system structures related to philly.
"""

import os


def app_dir_log_fns(app_dir):
    """
    Get the log files of an experiment for certain cluster.
    """
    log_dir = os.path.join(app_dir, 'logs')
    trial_nums = []
    for trial_dn in os.listdir(log_dir):
        try:
            trial_nums.append(int(trial_dn))
        except:
            continue
    trial_nums.sort()
    l_fn = []
    for trial_num in reversed(trial_nums):
        trial_dn = '{}'.format(trial_num)
        fn = os.path.join(log_dir, trial_dn, 'petridish_main', 'log.log')
        if os.path.exists(fn):
            l_fn.append(fn)
    return l_fn

def app_dir_stdout_fns(app_dir):
    """
    Get the log files of an experiment for certain cluster.
    """
    log_dir = os.path.join(app_dir, 'stdout')
    trial_nums = []
    for trial_dn in os.listdir(log_dir):
        try:
            trial_nums.append(int(trial_dn))
        except:
            continue
    trial_nums.sort()
    l_fn = []
    for trial_num in reversed(trial_nums):
        trial_dn = '{}'.format(trial_num)
        fn = os.path.join(log_dir, trial_dn, 'stdout.txt')
        if os.path.exists(fn):
            l_fn.append(fn)
    return l_fn


def cust_exp_app_dir(eidx, philly_log_root='./petridish_models_logs'):
    """
    Get app dir of a cust exp idx
    """
    l_app_dir = []
    for dn in os.listdir(philly_log_root):
        if 'petridish_{}_'.format(eidx) in dn:
            l_app_dir.append(os.path.join(philly_log_root, dn))
    return l_app_dir

def cust_exps_str_to_list(cust_exps_str):
    cust_exps = []
    if not cust_exps_str:
        return cust_exps

    intervals = cust_exps_str.strip().split(',')
    for interval in intervals:
        interval = interval.strip()
        if len(interval) == 0:
            continue
        minmax = interval.split('..')
        if len(minmax) == 1:
            cust_exps.append(minmax[0])
        else:
            for eidx in range(int(minmax[0]), int(minmax[1])+1):
                cust_exps.append(str(eidx))
    #print("The following eidx are to be analyzed {}".format(cust_exps))
    return cust_exps
