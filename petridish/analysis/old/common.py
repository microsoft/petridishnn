
import re
import os
import copy
import json

img_dir = './analysis/petridish_img/'
ann_models_logs = './petridish_models_logs'
experiment_list_fn = './petridish/cust_exps_gen/experiment_list.txt'
cache_dir = './temp/cache_perf'

def exp_dir_to_eidx(dn):
    reret = re.search(r'petridish_([0-9]*)_', dn)
    eidx = None
    if reret:
        try:
            eidx =  int(reret.group(1))
        except:
            pass
    return eidx

def eidx_to_exp_dir(eidx):
    return r'petridish_{}_'.format(eidx)

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
    print("The following eidx are to be analyzed {}".format(cust_exps))
    return cust_exps


class ExperimentRecord(object):
    def __init__(self, **kwargs):
        self.set_new(**kwargs)

    def set_new(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __repr__(self):
        selfdict = self.__dict__
        return json.dumps(selfdict)


def for_cust_exps(cust_exps, func_exp_dir, state):
    for dn in os.listdir(ann_models_logs):
        eidx = exp_dir_to_eidx(dn)
        if str(eidx) in cust_exps:
            exp_dir = os.path.join(ann_models_logs, dn)
            context = ExperimentRecord(eidx=eidx, exp_dir=exp_dir)
            print("Enterring exp_dir={}".format(exp_dir))
            state = func_exp_dir(exp_dir=exp_dir, state=state, context=context)
    return state


def for_trial_stdouts(exp_dir=None, func_stdout=None, state=[], context=None):
    stdout_dir = os.path.join(exp_dir, 'stdout')
    for dn in os.listdir(stdout_dir):
        tidx = None
        try:
            tidx = int(dn)
        except:
            continue
        if context:
            context = copy.deepcopy(context)
            context.set_new(exp_dir=exp_dir, tidx=tidx)
        else:
            context = ExperimentRecord(exp_dir=exp_dir, tidx=tidx)
        stdout_fn = os.path.join(stdout_dir, str(tidx), 'stdout.txt')
        print("Enterring log_fn={}".format(stdout_fn))
        state = func_stdout(log_fn=stdout_fn, state=state, context=context)
    return state


def filter_xy(xs, ys, args):
    indices = range(len(xs))

    def _pass_x_min(i):
        return args.plot_x_min is None or xs[i] >= args.plot_x_min
    def _pass_x_max(i):
        return args.plot_x_max is None or xs[i] <= args.plot_x_max
    def _pass_y_min(i):
        return args.plot_y_min is None or ys[i] >= args.plot_y_min
    def _pass_y_max(i):
        return args.plot_y_max is None or ys[i] <= args.plot_y_max
    def _pass_all(i):
        return _pass_x_min(i) and _pass_x_max(i) and _pass_y_max(i) and _pass_y_min(i)

    indices2 = [i for i in indices if _pass_all(i)]
    return indices2

    #print("Remaining indices are {}".format(list(indices2)))
    #print("inputs are {} {} ".format(xs,ys))
    #xs = [xs[i] for i in indices2]
    #ys = [ys[i] for i in indices2]
    #print("returns are {} {}".format(xs,ys))
    #return [xs, ys, indice
