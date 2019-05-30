import numpy as np
import re
import os
import bisect
from petridish.utils.geometry import _convex_hull_from_points
from functools import partial
from petridish.analysis.old.common import (
    img_dir, ann_models_logs, experiment_list_fn, cust_exps_str_to_list)

search_analysis_args = None

def grep_val_err_from_log(log_fn):
    min_ve = 2.0
    if os.path.exists(log_fn):
        with open(log_fn, 'rt') as fin:
            for li, line in enumerate(fin):
                line = line.strip()
                reret = re.search(r'val_err: ([0-9\.]*)$', line)
                if reret is not None:
                    ve = float(reret.group(1))
                    if min_ve > ve:
                        min_ve = ve
                        #min_li = li
    return min_ve


def grep_val_err_change_history_mp(
        log_root, stdout_fn=None, check_is_recover=False, get_all=False):
    l_li = []
    l_mi = []
    l_ve = []
    l_fp = []
    l_tc = [] # training cost
    curr_fp = np.float64(0)
    last_li = -1
    last_tc = 0
    min_ve = 2.0
    is_recover = False
    log_main = stdout_fn if stdout_fn else os.path.join(log_root, 'log.log')
    if os.path.exists(log_main):
        with open(log_main, 'rt') as fin:
            li = -1
            for line in fin:
                reret = re.search(r'mi=([0-9]*) val_err=([0-9\.]*)(.*)', line)
                if reret is None:
                    continue

                mi = int(reret.group(1))
                try:
                    ve = float(reret.group(2))
                except:
                    continue
                remainder_str = reret.group(3)

                # grep the line again for the recover.
                if check_is_recover and not is_recover:
                    reret = re.search(r'Recover: mi', line)
                    if reret is not None:
                        is_recover = True
                # grep flops
                reret = re.search(r'Gflops=([0-9\.]*)', remainder_str)
                if reret:
                    fp = float(reret.group(1))
                else:
                    fp = None
                # grep test error
                reret = re.search(r'test_err=([0-9\.]*)', remainder_str)
                if reret:
                    te = float(reret.group(1))
                else:
                    if not is_recover:
                        # older version has no separate val_err/test_err
                        te = ve
                    else:
                        # Previous runs has the te, current run report any value (2.0, inf)
                        te = 3.0
                # update number of model finished (li) and total training flops(curr_fp)
                li += 1
                if fp is not None:
                    curr_fp += (
                        fp *
                        search_analysis_args.max_train_model_epoch *
                        search_analysis_args.ds_train_size
                    )

                last_li = li
                last_tc = curr_fp
                err_value = ve if search_analysis_args.use_ve else te
                if err_value <= min_ve or (get_all and fp is not None):
                    min_ve = min(err_value, min_ve)
                    l_mi.append(mi)
                    l_ve.append(err_value)
                    l_li.append(li)
                    l_fp.append(fp)
                    l_tc.append(curr_fp)

    # Fake point in the end for plotting and info on the total trials
    l_li.append(last_li)
    l_ve.append(min_ve)
    l_mi.append(-1)
    l_fp.append(-1)
    l_tc.append(last_tc)
    if check_is_recover:
        return l_li, l_mi, l_ve, l_fp, l_tc, is_recover
    return l_li, l_mi, l_ve, l_fp, l_tc


def mi_to_perf(mi, log_root):
    log_fn = os.path.join(log_root, str(mi), 'log.log')
    return mi_log_to_perf(log_fn)

def mi_log_to_perf(log_fn):
    with open(log_fn, 'rt') as fin:
        multi_add_sum = 0
        ve = None
        for line in fin:
            line = line.strip()
            reret = re.search(r'multi-add.* ([0-9\.]*)$', line)
            if reret:
                multi_add = float(reret.group(1))
                multi_add_sum += multi_add
                continue

            reret = re.search(r'evaluation_value=\[([0-9\.]*)[,\]]', line)
            if reret:
                ve = float(reret.group(1))
    return multi_add_sum * 2, ve


def grep_val_err_change_history_from_app_dir_mp(app_dir, get_all=False):
    ll_li, ll_mi, ll_ve, ll_fp, ll_tc = [], [], [], [], []
    trials = os.listdir(os.path.join(app_dir, 'logs'))
    trials_int = []
    for trial in trials:
        try:
            trial_int = int(trial)
        except:
            continue
        trials_int.append(trial_int)
    trials_int.sort()

    for triali in trials_int:
        triali = str(triali)
        l_li, l_mi, l_ve, l_fp, l_tc = [], [], [], [], []
        trial_dn = os.path.join(app_dir, 'logs', triali)
        script_entries = os.listdir(trial_dn)
        if len(script_entries) > 0:
            log_root = os.path.join(trial_dn, script_entries[0])
            stdout_fn = os.path.join(app_dir, 'stdout', triali, 'stdout.txt')
            l_li, l_mi, l_ve, l_fp, l_tc, is_recover = \
                grep_val_err_change_history_mp(log_root, stdout_fn=stdout_fn, check_is_recover=True, get_all=get_all)
        if len(l_li) == 0:
            continue
        if is_recover:
            # Four cases:
            # 1. no gflops, no recover
            # 2. gflops , no recover
            # 3. gflops , recover
            # 4. no gflops, recover ( does not exist. )
            assert len(ll_mi) > 0, "{} has no predecessor".format(triali)
            del ll_li[-1][-1]
            del ll_mi[-1][-1]
            del ll_ve[-1][-1]
            del ll_fp[-1][-1]
            tc_offset = ll_tc[-1][-1]
            del ll_tc[-1][-1]
            min_ve = ll_ve[-1][-1] if len(ll_ve[-1]) > 0 else 2.0

            existing_mi = set(ll_mi[-1])
            for li, mi, ve, fp, tc in zip(l_li, l_mi, l_ve, l_fp, l_tc):
                if mi in existing_mi and fp is None:
                    continue
                if min_ve >= ve or (get_all and fp is not None):
                    ll_li[-1].append(li)
                    ll_mi[-1].append(mi)
                    ll_ve[-1].append(ve)
                    ll_fp[-1].append(fp)
                    ll_tc[-1].append(tc_offset + tc)
                    min_ve = ve
        else:
            ll_li.append(l_li)
            ll_mi.append(l_mi)
            ll_ve.append(l_ve)
            ll_fp.append(l_fp)
            ll_tc.append(l_tc)

    return ll_li, ll_mi, ll_ve, ll_fp, ll_tc


def grep_val_err_change_history_sp(stdout_fn):
    with open(stdout_fn, 'rt') as fin:
        min_ve = 2.0
        history_ve = []
        history_mi = []
        history_li = []
        mi = -1
        li = -1
        for line in fin:
            reret = re.search(r'mi=([0-9]*) pi=', line)
            if reret:
                mi = int(reret.group(1))

            reret = re.search(r'val_err: ([0-9\.]*)$', line)
            if reret:
                li += 1
                ve = float(reret.group(1))
                if min_ve >= ve:
                    min_ve = ve
                    history_mi.append(mi)
                    history_ve.append(ve)
                    history_li.append(li)

    return history_li, history_mi, history_ve


def grep_val_err_change_history_from_app_dir_sp(app_dir):

    stdout_dn = os.path.join(app_dir, 'stdout')
    if not os.path.exists(stdout_dn):
        print("No stdout dir is found")
        return [], [], []

    ll_li, ll_mi, ll_ve = [], [], []
    for triali in os.listdir(stdout_dn):
        stdout_fn = os.path.join(stdout_dn, triali, 'stdout.txt')
        if not os.path.exists(stdout_fn):
            print("stdout.txt doesn't exist, but its triali dir exists. The expect stdout is {}".format(stdout_fn))
            l_li, l_mi, l_ve = [], [], []
        else:
            l_li, l_mi, l_ve = grep_val_err_change_history_sp(stdout_fn)
        ll_li.append(l_li)
        ll_mi.append(l_mi)
        ll_ve.append(l_ve)

    return ll_li, ll_mi, ll_ve


def grep_val_err_change_history_from_cust_exps(cust_exps, version=None, get_all=False):
    if isinstance(cust_exps, int):
        cust_exps = str(cust_exps)

    if version == 'mp':
        grep_func = partial(grep_val_err_change_history_from_app_dir_mp, get_all=get_all)
    elif version == 'sp':
        grep_func = grep_val_err_change_history_from_app_dir_sp
    else:
        raise Exception("Version must be specified mp/sp")

    key = r'petridish_{}_'.format(cust_exps)
    l_app_dir, lll_li, lll_mi, lll_ve, lll_fp, lll_tc = [], [], [], [], [], []
    for dn in os.listdir(ann_models_logs):
        reret = re.search(key, dn)
        if reret is not None:
            app_dir = os.path.join(ann_models_logs, dn)

            ll_li, ll_mi, ll_ve, ll_fp, ll_tc = grep_func(app_dir)
            l_app_dir.append(app_dir)
            lll_li.append(ll_li)
            lll_mi.append(ll_mi)
            lll_ve.append(ll_ve)
            lll_fp.append(ll_fp)
            lll_tc.append(ll_tc)
    return l_app_dir, lll_li, lll_mi, lll_ve, lll_fp, lll_tc


def min_val_err_stdout(l_li, l_mi, l_ve):
    min_ve = 2.0
    min_li = -1
    min_mi = -1
    for li, mi, ve in zip(l_li, l_mi, l_ve):
        if ve < min_ve:
            min_ve = ve
            min_li = li
            min_mi = mi
    return min_li, min_mi, min_ve

def min_val_err_app_dir(ll_li, ll_mi, ll_ve):
    min_ve = 2.0
    min_li = -1
    min_mi = -1
    min_ti = '0'
    for ti, l_li, l_mi, l_ve in zip(range(len(ll_li)), ll_li, ll_mi, ll_ve):
        li, mi, ve = min_val_err_stdout(l_li, l_mi, l_ve)
        if ve < min_ve:
            min_ve = ve
            min_li = li
            min_mi = mi
            min_ti = str(ti + 1)
    return min_ti, min_li, min_mi, min_ve

def min_val_err_cust_exps(l_app_dir, lll_li, lll_mi, lll_ve):
    min_ve = 2.0
    min_li = -1
    min_mi = -1
    min_ti = '0'
    min_app = ''
    for app_dir, ll_li, ll_mi, ll_ve in zip(l_app_dir, lll_li, lll_mi, lll_ve):
        ti, li, mi, ve = min_val_err_app_dir(ll_li, ll_mi, ll_ve)
        if ve < min_ve:
            min_ve = ve
            min_li = li
            min_mi = mi
            min_ti = ti
            min_app = app_dir

    return min_app, min_ti, min_li, min_mi, min_ve

def min_val_err_cust_exps_v2(lll_ve):
    min_ve = 2.0
    min_ijk = (-1, -1, -1)
    for i, ll_ve in enumerate(lll_ve):
        for j, l_ve in enumerate(ll_ve):
            for k, ve in enumerate(l_ve):
                if ve < min_ve:
                    min_ijk = (i, j, k)
                    min_ve = ve
    return min_ve, min_ijk[0], min_ijk[1], min_ijk[2]


def _fuse_all(l_xs, l_ys, method='avg'):
    """
    Args:
    l_xs (list of list of x val)
    l_ys (list of list of y val)
    method (str) : 'avg' or 'min'

    Returns
    (xs, ys) : both xs and ys are list of val.
    They represent the value for ploting the merged
    results of inputs. Either by taking the average
    of the curves at each x value or the min.
    The x value are evenly spreaded at about 25 locations.
    """
    l_xs = list(filter(lambda xs : len(xs) >=2, l_xs))
    l_ys = list(filter(lambda ys : len(ys) >=2, l_ys))
    if len(l_xs) == 0:
        print("nothing to fuse")
        return [], []
    x_max = max([ xs[-1] for xs in l_xs ])
    x_min = min([ xs[0] for xs in l_xs ])

    step = (x_max - x_min) / 24.
    if step == 0:
        print("everything is at one point")
        return [x_min], [l_ys[0][-1]]

    plt_xs = np.arange(x_min, x_max + 2 * step, step, dtype=np.float32)
    plt_ys = np.zeros_like(plt_xs)
    plt_cnts = np.zeros_like(plt_xs)
    def _linear_interp_plot_val(_xs, _ys, x):
        idx = bisect.bisect_left(_xs, x)
        if idx == 0:
            return None #ys[0]
        if idx == len(_xs):
            return None
        x1, x2 = _xs[idx-1:idx+1]
        y1, y2 = _ys[idx-1:idx+1]
        y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
        return y

    for i, px in enumerate(plt_xs):
        for xs, ys in zip(l_xs, l_ys):
            py = _linear_interp_plot_val(xs, ys, px)
            if py is None:
                continue
            cnt = plt_cnts[i]
            if cnt == 0:
                plt_ys[i] = py
            else:
                if method == 'avg':
                    plt_ys[i] = (plt_ys[i] * cnt + py) / (cnt + 1.)
                elif method == 'min':
                    plt_ys[i] = min(plt_ys[i], py)
            plt_cnts[i] += 1.0

    indices = np.nonzero(plt_cnts)[0]
    return list(plt_xs[indices]), list(plt_ys[indices])


def _ensure_decrease(xs, ys):
    for i in range(1, len(ys)):
        if ys[i] > ys[i-1]:
            ys[i] = ys[i-1]
    return xs, ys


def load_baseline_dict(baseline_fn=None):
    if baseline_fn is None:
        baseline_fn = './data/openml/baseline/oaa_loss.txt'
    baseline_dict = dict()
    with open(baseline_fn, 'rt') as fin:
        for line in fin:
            try:
                idx, val = line.strip().split()
            except:
                raise ValueError("Invalid line in baselin_fn {}: {}".format(
                    baseline_fn, line.strip()
                ))
            try:
                idx = int(idx)
                name = 'openml_{}'.format(idx)
            except:
                name = str(idx)
            try:
                val = float(val)
            except:
                raise ValueError("Invalid value in baseline_fn {}: {}".format(
                    baseline_fn, line.strip()
                ))
            baseline_dict[name] = val
    return baseline_dict

def desc_to_ds_name(desc):
    reret = re.search(r'^[0-9]* : ([a-zA-Z0-9_]*)', desc)
    assert reret, "Cannot find ds_name from desc : {}".format(desc)
    return reret.group(1)


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

    indices2 = list(filter(lambda i : _pass_all(i), indices))
    #print("Remaining indices are {}".format(list(indices2)))
    #print("inputs are {} {} ".format(xs,ys))
    xs = [xs[i] for i in indices2]
    ys = [ys[i] for i in indices2]
    #print("returns are {} {}".format(xs,ys))
    return xs, ys

def fuse_progress(lll_li, lll_ve, fuse_method='avg'):
    l_xs, l_ys = [], []
    for ll_li, ll_ve in zip(lll_li, lll_ve):
        l_xs.extend(ll_li)
        l_ys.extend(ll_ve)
    fused_li, fused_ve = _fuse_all(l_xs, l_ys, fuse_method)
    fused_li, fused_ve = _ensure_decrease(fused_li, fused_ve)
    return fused_li, fused_ve


def plot_progress(args, eidx_to_desc, eidx_to_ret):
    import matplotlib.pyplot as plt
    fontsize = 13
    plt.close('all')
    if args.compare_baseline:
        baseline_to_plot = set() # ds_name to xs
        max_xs = []
    y_multiplier = 100
    for eidx in eidx_to_desc:
        desc = eidx_to_desc[eidx]
        ret_dict = eidx_to_ret[eidx]
        lll_xs, lll_ys = [], []
        for app_id in ret_dict:
            app_dir, ll_li, ll_mi, ll_ve, ll_fp, ll_tc = ret_dict[app_id]
            if args.plot_tc:
                lll_xs.append(ll_tc)
            else:
                lll_xs.append(ll_li)
            lll_ys.append(ll_ve)

        if args.plot_type == 'merge_min':
            fuse_method = 'min'
        elif args.plot_type == 'merge_avg':
            fuse_method = 'avg'
        elif not args.plot_type:
            fuse_method = 'avg'
        xs, ys = fuse_progress(lll_xs, lll_ys, fuse_method)
        if len(xs) <= 1:
            continue
        xs, ys = filter_xy(xs, ys, args)
        if len(xs) == 0:
            print("Nothing to plot with args {}".format(args))

        ys = list(map(lambda _y : _y * y_multiplier, ys))
        plt.plot(xs, ys, label=desc)

        if args.compare_baseline:
            ds_name = desc_to_ds_name(desc)
            baseline_to_plot.add(ds_name)
            if len(max_xs) == 0 or (len(xs) > 0 and xs[-1] > max_xs[-1]):
                max_xs = xs

    if args.compare_baseline:
        for ds_name in baseline_to_plot:
            plt.plot(
                max_xs,
                [args.baseline_dict[ds_name] * y_multiplier ] * len(max_xs),
                label='{}, baseline'.format(ds_name)
            )

    plt.legend(fontsize=fontsize * 2 // 3)
    xlabel = (
        'Number of models trained' if not args.plot_tc else
        'GFLOPs during Forward in Training')
    tc_post_fix = "_tc" if args.plot_tc else ""
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel('Validation error', fontsize=fontsize)
    plt.savefig(
        os.path.join(
            img_dir,
            'cust_exps_{}{}.png'.format(
                '_'.join(args.cust_exps), tc_post_fix)
        ),
        dpi=plt.gcf().dpi, bbox_inches='tight'
    )

def plot_convex_hull_app(ret_dict, app_id, desc, do_plot=True):
    app_dir, ll_li, ll_mi, ll_ve, ll_fp, ll_tc = ret_dict[app_id]
    ll_ch_mi = []
    trial = -1
    for l_li, l_mi, l_ve, l_fp in zip(ll_li, ll_mi, ll_ve, ll_fp):
        trial += 1
        indices, eps_indices = _convex_hull_from_points(
            l_fp[:-1], l_ve[:-1], eps=0)
        if len(indices) < 2:
            continue
        xs = [l_fp[i] for i in indices]
        ys = [l_ve[i] for i in indices]
        l_ch_mi = [l_mi[i] for i in indices]
        if do_plot:
            label = desc + '|' + app_id + '|' + str(trial)
            print("label : {}\nhull_mi : {}".format(label, l_ch_mi))
            plt.plot(
                xs, [_y * 100 for _y in ys],
                label=label)
        ll_ch_mi.append(l_ch_mi)
    return ll_ch_mi

def plot_convex_hull(args, eidx_to_desc, eidx_to_ret, eidx_to_loc):
    import matplotlib.pyplot as plt
    fontsize = 13
    plt.close('all')

    for eidx in eidx_to_desc:
        desc = eidx_to_desc[eidx]
        ret_dict = eidx_to_ret[eidx]
        if args.plot_type == 'each':
            for app_id in ret_dict:
                ll_ch_mi = plot_convex_hull_app(ret_dict, app_id, desc)

        elif args.plot_type == 'best':
            app_dir = eidx_to_loc[eidx][0]
            app_id = app_dir_to_app_id(app_dir)
            ll_ch_mi = plot_convex_hull_app(ret_dict, app_id, desc)

        elif args.plot_type == 'merge_min':
            raise NotImplementedError('merge_min of plot_ch')

        elif args.plot_type == 'merge_avg':
            raise NotImplementedError('merge_avg of plot_ch')

        # l_xs = []
        # l_ys = []
        # for app_id in ret_dict:
        #     app_dir, ll_li, ll_mi, ll_ve, ll_fp, ll_tc = ret_dict[app_id]
        #     for l_li, l_mi, l_ve, l_fp in zip(ll_li, ll_mi, ll_ve, ll_fp):
        #         indices, eps_indices = _convex_hull_from_points(
        #             l_fp[:-1], l_ve[:-1], eps=0)
        #         xs = [ l_fp[i] for i in indices]
        #         ys = [ l_ve[i] for i in indices]
        #         l_xs.append(xs)
        #         l_ys.append(ys)
        # print("n_to_fuse={}".format(len(l_xs)))
        # # select the best final performance
        # fused_xs, fused_ys = None, None
        # for xs, ys in zip(l_xs, l_ys):
        #     if len(ys) == 0:
        #         continue
        #     if fused_ys is None or fused_ys[-1] > ys[-1]:
        #         # keep the app that has the best final result
        #         fused_xs = xs
        #         fused_ys = ys
        # fused_ys = list(map(lambda _y : _y * 100, fused_ys))
        # plt.plot(fused_xs[:], fused_ys[:], label=desc)

    plt.legend(fontsize=fontsize * 2 // 3)
    plt.xlabel('GFLOPS', fontsize=fontsize)
    plt.ylabel('Validation error', fontsize=fontsize)
    plt.savefig(
        os.path.join(img_dir, 'cust_exps_{}_final_CH.png'.format('_'.join(args.cust_exps))),
        dpi=plt.gcf().dpi, bbox_inches='tight')

def app_dir_to_app_id(app_dir):
    app_id = os.path.basename(os.path.normpath(app_dir))
    app_id = 'application' + app_id.split('application')[1]
    return app_id

def get_search_results(args):
    eidx_to_ret = dict()
    eidx_to_loc = dict()
    for eidx in args.cust_exps:
        ret = grep_val_err_change_history_from_cust_exps(
            eidx, args.version, get_all=args.plot_ch)
        l_app_dir, lll_li, lll_mi, lll_ve, lll_fp, lll_tc = ret
        ret_dict = dict()
        for (app_dir, ll_li, ll_mi, ll_ve, ll_fp, ll_tc) in \
                zip(l_app_dir, lll_li, lll_mi, lll_ve, lll_fp, lll_tc):
            app_id = app_dir_to_app_id(app_dir)
            ret_dict[app_id] = [app_dir, ll_li, ll_mi, ll_ve, ll_fp, ll_tc]
        eidx_to_ret[int(eidx)] = ret_dict
        print("eidx={} :\n app_dir={}\n li={}\n mi={}\n ve={}\n Gfp={}\n TC={}\n".format(
                eidx, l_app_dir, lll_li, lll_mi, lll_ve, lll_fp, lll_tc))

        # min_i : best app idx
        # min_j : best trial idx
        # min_k : best val_err idx
        min_ve, min_i, min_j, min_k = min_val_err_cust_exps_v2(lll_ve)
        min_loc = (l_app_dir[min_i], # app_dir str
            min_j, # trial idx ; always 0 for ones with recover
            lll_li[min_i][min_j][min_k],
            lll_mi[min_i][min_j][min_k],
            lll_tc[min_i][min_j][min_k],
            lll_fp[min_i][min_j][min_k],
            min_ve
        )
        eidx_to_loc[int(eidx)] = min_loc
        print(min_loc)
    return eidx_to_ret, eidx_to_loc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--plot_ch', default=False, action='store_true')
    parser.add_argument('--plot_type', type=str, default=None,
        choices=['each', 'best', 'merge_min', 'merge_avg'],
        help=(
            'How the different app_id of the same eidx interact:'
            ' each: each app has its own curve on the plot; '
            ' best: only the one that has the lowest final y val is shown; '
            ' merge_min: merge the hulls by taking min at each x val; '
            ' merge_avg: merge the hulls by taking avg at each x val'
        )
    )
    parser.add_argument('--plot_tc', default=False, action='store_true')
    parser.add_argument('--plot_x_min', default=None, type=int)
    parser.add_argument('--plot_x_max', default=None, type=int)
    parser.add_argument('--plot_y_min', default=None, type=int)
    parser.add_argument('--plot_y_max', default=None, type=int)
    parser.add_argument('--stdout_fn', type=str, default=None)
    #parser.add_argument('--log_root', type=str, default=None)
    #parser.add_argument('--app_dir', type=str, default=None)
    parser.add_argument('--cust_exps', type=str, default=None)
    parser.add_argument('--version', type=str, default='mp', choices=['mp'])
    parser.add_argument('--max_train_model_epoch', type=int, default=80)
    parser.add_argument('--ds_train_size', type=int, default=50000)
    parser.add_argument('--compare_baseline', default=False, action='store_true')
    parser.add_argument('--baseline_fn', default=None, type=str)
    parser.add_argument('--use_ve', default=False, action='store_true',
        help='Use validation error for plotting (False/default) or use test error (True)')
    args = parser.parse_args()

    args.cust_exps = cust_exps_str_to_list(args.cust_exps)

    if args.plot_tc:
        args.plot = True
    assert not (args.plot and args.plot_ch), \
        'cannot plot  and plot_ch together as they need different points'

    args.baseline_dict = None
    if args.compare_baseline:
        args.baseline_dict = load_baseline_dict(args.baseline_fn)

    search_analysis_args = args

    eidx_to_desc = dict()
    with open(experiment_list_fn, 'rt') as fin:
        for li, line in enumerate(fin):
            line = line.strip()
            try:
                eidx = int(line.split(':')[0].strip())
            except:
                eidx = None
            if eidx in args.cust_exps:
                # This checks that the line is an experiment
                eidx_to_desc[eidx] = line
    eidx_to_ret, eidx_to_loc = get_search_results(args)

    if args.plot:
        plot_progress(args, eidx_to_desc, eidx_to_ret)

    if args.plot_ch:
        plot_convex_hull(args, eidx_to_desc, eidx_to_ret, eidx_to_loc)


    # Almost never used. comment out for now. in case for some niche usage.
    # elif args.app_dir is not None:
    #     if args.version == 'sp':
    #         grep_func = grep_val_err_change_history_from_app_dir_sp
    #     else:
    #         grep_func = grep_val_err_change_history_from_app_dir_mp
    #     ll_li, ll_mi, ll_ve, ll_fp = grep_func(args.app_dir)
    #     print(ll_li, ll_mi, ll_ve, ll_fp)
    #     if args.find_min:
    #         min_loc = min_val_err_app_dir(ll_li, ll_mi, ll_ve)
    #         print(min_loc)

    # elif args.log_root is not None:
    #     l_li, l_mi, l_ve, l_fp, l_tc = \
    #         grep_val_err_change_history_mp(args.log_root, args.stdout_fn)
    #     print(l_li, l_mi, l_ve, l_fp, l_tc)
    #     if args.find_min:
    #         min_loc = min_val_err_stdout(l_li, l_mi, l_ve)
    #         print(min_loc)

    # elif args.stdout_fn is not None:
    #     l_li, l_mi, l_ve = grep_val_err_change_history_sp(args.stdout_fn)
    #     print(l_li, l_mi, l_ve)
    #     if args.find_min:
    #         min_loc = min_val_err_stdout(l_li, l_mi, l_ve)
    #         print(min_loc)




