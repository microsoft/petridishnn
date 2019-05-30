import numpy as np
import re
import os
import bisect
from petridish.utils.geometry import _convex_hull_from_points
from functools import partial
import copy
import subprocess
from tensorpack.utils.serialize import loads, dumps

from petridish.analysis.old.common import (
    img_dir, ann_models_logs, experiment_list_fn, exp_dir_to_eidx,
    for_cust_exps, for_trial_stdouts, cust_exps_str_to_list,
    ExperimentRecord, cache_dir, filter_xy)

INCLUDE_AUX_MA = False
REQUIRED_EPOCH = 500
FORCE_LOAD = False

def multi_add_from_log(log_fn):
    multi_add = 0.0
    n_params = -1.0
    with open(log_fn, 'rt') as fin:
        for line in fin:
            reret = re.search(r'(.*)multi-add.* ([0-9\.]*)$', line.strip())
            if reret:
                try:
                    prefix_reret = re.search(r'aux_preprocess', reret.group(1))
                    if not prefix_reret or INCLUDE_AUX_MA:
                        multi_add += float(reret.group(2))
                    continue
                except:
                    pass

            reret = re.search(r'#params=([0-9]*),', line)
            if reret:
                n_params = float(reret.group(1))
                break
    return multi_add, n_params


def val_err_from_log(log_fn):
    def tail(f, n):
        cmd = "egrep \"Epoch ([0-9]*)|val_err: ([0-9\\.]*)$\" {}".format(f)
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE)
        lines = proc.stdout.readlines()
        return [line.decode('utf-8') for line in lines]

    lines = tail(log_fn, 100)
    min_ve_epoch = -1
    epoch = -1
    min_ve = 2.0
    for line in lines:
        reret = re.search(r'Epoch ([0-9]*)|val_err: ([0-9\.]*)', line)
        if reret:
            if reret.group(1) is not None:
                try:
                    new_epoch = int(reret.group(1))
                    if min_ve_epoch == -1 and min_ve < 1.0:
                        min_ve_epoch = new_epoch - 1
                    epoch = new_epoch
                except:
                    pass
            elif reret.group(2) is not None:
                try:
                    ve = float(reret.group(2))
                    if ve <= min_ve:
                        min_ve = ve
                        min_ve_epoch = epoch
                except:
                    pass
    return min_ve, min_ve_epoch


def perf_from_log(log_fn):
    """
    Args:
    log_fn : a stdout file xxx/stdout/triali/stdout.txt
    """
    dn = os.path.dirname(log_fn)
    cache_fn = dn.replace('/', '__')
    cache_fn = os.path.join(cache_dir, cache_fn)
    if os.path.exists(cache_fn):
        with open(cache_fn, 'rb') as fin:
            ss = fin.read()
        try:
            ret = loads(ss)
        except:
            pass
        if ret and not FORCE_LOAD:
            return ret

    if os.path.exists(log_fn):
        min_ve, min_ve_epoch = val_err_from_log(log_fn)
        multi_add, n_params = multi_add_from_log(log_fn)
        ret = (min_ve, multi_add * 2. * 1e-9, min_ve_epoch)
        with open(cache_fn, 'wb') as fout:
            fout.write(dumps(ret))
        return ret
    else:
        return 2.0, -1.0, -1


def init_state_for_model_perf():
    # min_ve, multi_add, epoch when min_ve
    return []


def func_stdout_for_model_perf(log_fn, state, context):
    if context is not None:
        record = copy.deepcopy(context)
    else:
        context = ExperimentRecord()
    ve, fp, ep = perf_from_log(log_fn)
    record.set_new(ve=ve, fp=fp, ep=ep)
    if ve < 1.0:
        print(record)
    state.append(record)
    return state


def merge_state(state, required_epoch=REQUIRED_EPOCH):
    state = [x for x in state if x.ep > required_epoch]
    state.sort(key=lambda x : x.eidx)
    cnt = 0.0
    avg_ve = 0.0
    min_ve = 2.0
    ret = dict()
    for idx, x in enumerate(state):
        avg_ve += x.ve
        min_ve = min(min_ve, x.ve)
        cnt += 1.0
        if idx == len(state) - 1 or x.eidx != state[idx+1].eidx:
            avg_ve /= cnt + int(cnt < 1)
            ret[x.eidx] = (x.fp, avg_ve, min_ve)
            cnt = 0.0
            avg_ve = 0.0
            min_ve = 2.0
    return ret

def amoeba_A_scatter_xys():
    def xys_to_xs_ys(xys, name):
        return xys[::2], xys[1::2], name

    amoeba_a_xys = xys_to_xs_ys(
        [
            0.8243604811532328, 3.717231330127244,
            0.8561252293632995, 3.576010697624216,
            0.9252184498039169, 3.4887507045800703,
            0.9434581389491863, 3.4586750296823094,
            0.9624620126404664, 3.6313001451135123
        ],
        name='Amoeba-A')

    amoeba_rl_xys = xys_to_xs_ys(
        [
            1.092913063813967, 3.527370087427893,
            1.1475961526929952, 3.474680690308576,
            1.169610712015639, 3.5061571303503114,
            1.2213824160800164, 3.491012556516316,
            1.2270612714821967, 3.566073420241536,
        ],
        name='Amoeba-RL')

    amoeba_rand_xys = xys_to_xs_ys(
        [
            0.950659606874303, 3.9451434944772914,
            0.9871001283235192, 3.925532782461653,
            0.9842045740738521, 3.94656104961443,
            0.9486779079668519, 4.012716021251334,
            0.9352593454301237, 4.012749601237663,
        ],
        name='Amoeba-Rand')

    return [amoeba_a_xys, amoeba_rl_xys, amoeba_rand_xys]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--find_min', default=False, action='store_true')
    parser.add_argument('--stdout_fn', type=str, default=None)
    parser.add_argument('--log_root', type=str, default=None)
    parser.add_argument('--app_dir', type=str, default=None)
    parser.add_argument('--cust_exps', type=str, default=None)
    parser.add_argument('--include_aux_ma', default=False, action='store_true')
    parser.add_argument('--force_load', default=False, action='store_true')
    parser.add_argument('--required_epochs', default=REQUIRED_EPOCH, type=int)
    parser.add_argument('--plot_scatter', default=False, action='store_true')
    parser.add_argument('--plot_x_min', default=None, type=float)
    parser.add_argument('--plot_x_max', default=None, type=float)
    parser.add_argument('--plot_y_min', default=None, type=float)
    parser.add_argument('--plot_y_max', default=None, type=float)
    args = parser.parse_args()

    args.cust_exps = cust_exps_str_to_list(args.cust_exps)
    FORCE_LOAD = args.force_load
    REQUIRED_EPOCH = args.required_epochs
    INCLUDE_AUX_MA = args.include_aux_ma

    if args.cust_exps:
        func_exp_dir_for_model_perf = partial(
            for_trial_stdouts,
            func_stdout=func_stdout_for_model_perf
        )

        state = for_cust_exps(
            args.cust_exps,
            func_exp_dir_for_model_perf,
            init_state_for_model_perf())

        ret_dict = merge_state(state)

        #
        if args.plot_scatter:
            import matplotlib.pyplot as plt
            plt.close('all')

            fig, ax = plt.subplots()
            eindices = list(ret_dict.keys())
            fp_idx, ve_idx, min_ve_idx = 0, 1, 2
            xs = [ret_dict[eidx][fp_idx] for eidx in eindices]
            y_multiplier = 100.
            ys = [ret_dict[eidx][ve_idx] * y_multiplier for eidx in eindices]
            remain_indices = filter_xy(xs, ys, args)
            xs = [xs[i] for i in remain_indices]
            ys = [ys[i] for i in remain_indices]
            eindices = [eindices[i] for i in remain_indices]

            ax.scatter(xs, ys, label='Ours')
            for i, eidx in enumerate(eindices):
                ax.annotate(str(eidx), (xs[i], ys[i]))

            baselines = amoeba_A_scatter_xys()
            for baseline in baselines:
                _xs, _ys, _name = baseline
                ax.scatter(_xs, _ys, label=_name)

            plt.grid()
            plt.xlabel('GFLOPS')
            plt.ylabel('Test Error')
            plt.legend()
            plt.savefig(
                os.path.join(
                    img_dir,
                    'cust_exps_{}_scatter.png'.format(
                        '_'.join(args.cust_exps))
                ),
                dpi=plt.gcf().dpi, bbox_inches='tight'
            )

        print(ret_dict)
        with open('./temp/model_analysis_ret.bin', 'wb') as fout:
            fout.write(dumps(ret_dict))







