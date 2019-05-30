import numpy as np
import re
import os

img_dir = './analysis/petridish_img/'
ann_models_logs = './petridish_models_logs'

def grep_val_err_from_log(log_fn):
    min_ve = 2.0
    with open(log_fn, 'rt') as fin:
        for line in fin:
            line = line.strip()
            reret = re.search(r'val_err: ([0-9\.]*)$', line)
            if reret is not None:
                min_ve = min(min_ve, float(reret.group(1)))
    return min_ve

def grep_val_err(logdir):

    l_ii = []
    local_val_errs = []
    remote_val_errs = []
    for triali in range(1, 10):
        exp_log_dir = os.path.join(logdir, 'logs', str(triali), 'petridish_main')
        if not os.path.exists(exp_log_dir):
            break
        max_ii = -1
        for dn in os.listdir(exp_log_dir):
            try:
                ii = int(dn)
                max_ii = max(ii, max_ii)
            except:
                continue

        for ii in range(max_ii+1):
            local_fn = os.path.join(exp_log_dir, str(ii), 'log.log')
            remote_fn = os.path.join(exp_log_dir, '{}_r'.format(str(ii)), 'log.log')

            if os.path.exists(local_fn) and os.path.exists(remote_fn):
                ve_local = grep_val_err_from_log(local_fn)
                ve_remote = grep_val_err_from_log(remote_fn)
                if ve_local < 1 and ve_remote < 1:
                    l_ii.append(ii)
                    local_val_errs.append(ve_local)
                    remote_val_errs.append(ve_remote)

    return l_ii, local_val_errs, remote_val_errs


def average_over_data_set(ds_name):
    if ds_name == 'cifar10':
        key = 'petridish_1_'
    elif ds_name == 'cifar100':
        key = 'petridish_2_'

    pattern_str = r'.*{}.*'.format(key)

    max_growth = 40
    cnts = [0.0] * max_growth
    local_ve = [0.0] * max_growth
    remote_ve = [0.0] * max_growth


    for dn in os.listdir(ann_models_logs):
        reret = re.search(pattern_str, dn)
        if reret is not None:
            exp_log_dir = os.path.join(ann_models_logs, dn)
            ii, local, remote = grep_val_err(exp_log_dir)
            for i, lve, rve in zip(ii, local, remote):
                cnts[i] +=1.
                local_ve[i] += lve
                remote_ve[i] += rve
    l_ii = []
    local_val_errs = []
    remote_val_errs = []
    for i in range(max_growth):
        if cnts[i] > 0:
            l_ii.append(i)
            local_val_errs.append(local_ve[i] / cnts[i])
            remote_val_errs.append(remote_ve[i] / cnts[i])

            print("ii = {} ; cnt = {}".format(i, cnts[i]))
    return l_ii, local_val_errs, remote_val_errs


def plot_inc_vs_scratch(l_ii, local_val_errs, remote_val_errs, times_100=True, ensure_decreasing=True, postfix=""):
    import matplotlib.pyplot as plt
    plt.close('all')
    if times_100:
        local_val_errs = list(map(lambda x : x * 100, local_val_errs))
        remote_val_errs = list(map(lambda x :  x* 100, remote_val_errs))
    if ensure_decreasing:
        for k in range(1, len(l_ii)):
            local_val_errs[k] = min(local_val_errs[k-1], local_val_errs[k])
            remote_val_errs[k] = min(remote_val_errs[k-1], remote_val_errs[k])
    plt.plot(l_ii, local_val_errs, label='Incremental')
    plt.plot(l_ii, remote_val_errs, label='From-scratch')
    plt.legend()
    fontsize=13
    plt.xlabel('Model index', fontsize=fontsize)
    plt.ylabel('Error Rate', fontsize=fontsize)

    plt.savefig(os.path.join(img_dir, 'inc_vs_scratch{}.png'.format('_{}'.format(postfix) if postfix else "" )),
        dpi=plt.gcf().dpi, bbox_inches='tight')


if __name__ == '__main__':
    for ds_name in ['cifar10', 'cifar100']:
        ret = average_over_data_set(ds_name)
        plot_inc_vs_scratch(ret[0], ret[1], ret[2], times_100=True, ensure_decreasing=True, postfix=ds_name)
