

import numpy as np
import json
import os, re, sys
import argparse
import subprocess


grep_info_cmd = \
r"""cat {in_fn} | grep -Pzo 'mi={mi} .*\n.*LayerInfoList.*\n.*\n'"""

grep_perf_cmd = \
r"""cat {in_fn} | grep -Po '\w+ : mi={mi} val_err=.* test_err=.* Gflops=.*'"""


def verb_print(ss, level):
    if level:
        print(ss)

def get_cmd_output_as_lines(cmd):
    #print("cmd is: \n" + cmd +'\n')
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    lines = [line.decode('utf-8') for line in lines]
    return lines

def inspect_one_model(in_fn, model_iter, out_fn=None, do_save=False, verbose=True):
    """
    This is a stand-alone function to inspect a log file `in_fn`
    in order to find a model with the given identitfication, `model_iter`.
    The model info will be saved to `out_fn` if `do_save` is True.

    Highly recommend to run this with do_save=False first to check
    the print out before a second run to save to `out_fn`.

    Args:

    in_fn (str) : log file path
    model_iter (int) : model id.
    out_fn (str) : file path to save the model if it is found.
        If it is None, a default one will be generated for you based on
        the other inputs.
    default one wi
    do_save (bool) : whether to save
    verbose (bool) : whether to print on screen some useful info

    Returns:

    A tuple that contains the following
    net_info_str: the net_info (basically a json obj) of the architecture
    ve : val error during search
    te : test error during search, if has one
    fp : flops of the model
    pi : parent model id.
    sd : search depth, i.e., the distance from the original seed model
    out_fn : the out_fn to save the net_info_str on.
        Useful if `out_fn=None` in the input
    """
    cmd = grep_info_cmd.format(in_fn=in_fn, mi=model_iter)
    lines = get_cmd_output_as_lines(cmd)
    net_info_str, ve, te, fp, pi, sd, out_fn = [None] * 7
    verb_print('-'*20 + '\nInspecting mi={}'.format(model_iter), verbose)
    if not lines:
        verb_print("Could not find mi={} in log_fn={}".format(model_iter, in_fn), verbose)
        return net_info_str, ve, te, fp, pi, sd, out_fn

    for line in lines[2::4]:
        net_info_str = line.strip()
        # remove random crap that philly inserted on stdout.
        net_info_str = net_info_str[net_info_str.find('{'):]
        net_info_str = net_info_str.replace("[1,0]<stdout>:", "")
        verb_print("net_info_str is \n" + net_info_str + "\n", verbose)

    if not out_fn:
        reret = re.search(r'([0-9]*_application_[0-9]*_[0-9]*).*/([0-9]*)/.*', in_fn)
        if reret:
            fn = reret.group(1) + '_' + reret.group(2) + '_{}.txt'.format(model_iter)
        else:
            fn = os.path.split(in_fn)[0].replace(os.sep, '_') + '_{}.txt'.format(model_iter)

        dn = 'result_net_info'
        if not os.path.exists(dn):
            os.makedirs(dn)
        out_fn = os.path.join(dn, fn)

    if do_save:
        with open(out_fn, 'wt') as fout:
            fout.write(net_info_str)

    for line in lines[::4]:
        reret = re.search(
            r'mi=([0-9]*) pi=([0-9]*) sd=([0-9]*)', line.strip()
        )
        if reret:
            mi, pi, sd = reret.group(1), reret.group(2), reret.group(3)
        else:
            print('ss={} does not have mi, pi, sd'.format(line))
        verb_print("mi={} pi={} sd={}".format(mi, pi, sd), verbose)

    cmd = grep_perf_cmd.format(in_fn=in_fn, mi=model_iter)
    lines = get_cmd_output_as_lines(cmd)
    for line in lines:
        reret = re.search(
            r'(\w+) : mi=[0-9]* val_err=(.*) test_err=(.*) Gflops=(.*)',
            line.strip())
        if reret:
            remote_type = reret.group(1)
            ve = reret.group(2)
            te = reret.group(3)
            fp = reret.group(4)
        verb_print("Remote type is {}".format(remote_type), verbose)
        verb_print("Perf during search is ve={} te={} fp={}".format(ve, te, fp), verbose)
    return net_info_str, ve, te, fp, pi, sd, out_fn


def inspect_models(l_in_fn, l_mi, l_out_fn=None, do_save=False, verbose=True):
    """
    This is a stand-alone function to inspect a list of models
    given in `l_mi` in the list of search logs `l_in_fn`.

    We assume the logs are from newest to the oldest of the
    same run. There are several of them because there were
    some interruptions and recovery during search.

    Args:
    l_in_fn (list of str) : list of input filenames
    l_mi (list of int) : list of model iter
    l_out_fn (list of str) : list of file path to save the models.
        None if you want this function to figure out something for you,
        based on other inputs.
    do_save (bool) : whether to save
    verbose (bool) : whether to print on screen some useful info

    Returns:
    A list of tuples. Each tuple is a result of `inspect_one_model`
    """
    l_ret = [None] * len(l_mi)
    in_fn_idx = 0
    mi_idx = len(l_mi) - 1
    for mi_idx, mi in enumerate(l_mi):
        # we have to loop over all to find one, since
        # we are not sure where each one is finished.
        for in_fn in l_in_fn:
            out_fn = None if not l_out_fn else l_out_fn[mi_idx]
            ret = inspect_one_model(in_fn, mi, out_fn, do_save, verbose)
            if ret[0]:
                l_ret[mi_idx] = ret
                break

    #verb_print(('-'*80 + '\nThe list of returns are {}'.format(l_ret)), verbose)
    return l_ret


def convex_hull_model_iters(l_in_fn, max_mi=None, verbose=True):
    """
    This is a stand-alone function to inspect a
    number of logs to find the identifications of models
    that are on the performance convex hull during search.

    We assume the logs are from newest to the oldest of the
    same run. There are several of them because there were
    some interruptions and recovery during search.

    Args:
    l_in_fn (list of str) : list of input filenames
    max_mi (int) : cut off the search so that the maximum
        Id. <= max_mmi

    Returns:
    A list of model Id. on the convex hull. Or None if nothing was found.
    """
    merged_lines = []
    for fn in l_in_fn:
        cmd = r"""cat {fn} | grep "l_mi=" """.format(fn=fn)
        lines = get_cmd_output_as_lines(cmd)
        if lines:
            if max_mi is None:
                # we only want the last line
                merged_lines = lines
                break
            else:
                # we will need to filter the lines
                merged_lines.extend(reversed(lines))
    ll_mi = [
        [int(s.strip()) \
            for s in line.split('l_mi=[')[1].split(']')[0].strip().split(',')] \
                for line in merged_lines
    ]
    if not ll_mi:
        verb_print("Could not find a file that has l_mi among logs={}".format(l_in_fn), verbose)
        return []

    if max_mi is None:
        l_mi = ll_mi[-1]
    else:
        for l_mi in ll_mi:
            local_max_mi = max(l_mi)
            if local_max_mi <= max_mi:
                break
    verb_print("The convex hull is l_mi={}".format(l_mi), verbose)
    return l_mi

