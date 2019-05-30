"""
The actors:
        Server, Sleeper, Remote, Crawler.

The play:
    1. Server has a new job.
	2. Server spawns a sleeper, which writes down the auto-script of the job and gets
        in a while-sleep-read loop to check at a specific location associated
        with the auto-script.
	3. Crawler crawls for auto-scripts. When crawler finds an auto-script,
        crawler launches it with a subprocess.
	4. The launched job by Crawler is a Remote.
	5. Remote will finish and write a special "mark_finished" file at the location
        where the sleeper is reading from.
	6. Sleeper finds the mark, and stops while-loop.
	7. Sleeper sends a message, defined by msg_func(), to Server, and terminates.
	8. Server receives the message. Server updates the queues and other states.
    9. Server looks into its queues for new job. (step 1)

The jobs:
    1. model training. This includes model w/ hallu and model w/o hallu.
    2. critic training. Each of the three queues q_child, q_parent, q_hallu has a critic.
       FIXME: remove all logics of critic as it is now never used.

The queues:
    1. q_parent: contains models that are trained. We add hallu to its peaked member.
        The new model is added to q_hallu.
    2. q_hallu: contains models that have newly added hallu. We are to initialize the hallu,
        Initialzied result is add result to q_child.
    3. q_child: contains models w/ initialized hallu. We are to train the full network.
        Trained result is added to q_parent, if it is not already too deep.

See the entrance functions:
    Server: __main__, server_main
    Remote: __main__, train_child, crawl_data_and_critic_train
    Sleeper: fork_and_train_model, fork_and_train_critic
    crawler: app/local_crawler.py
"""

import numpy as np
import tensorflow as tf
import os
import argparse
import copy
import re
import subprocess
import atexit
import itertools
import traceback
from functools import partial

from tensorpack.utils import logger, utils, fs, stats
from tensorpack.utils.serialize import loads, dumps

import anytime_models.examples.ann_app_utils as ann_app_utils

from petridish.info import (
    LayerInfo, LayerInfoList, CellNetworkInfo, net_info_from_str)
from petridish.model import (
    RecognitionModel, MLPModel, NUM_STATS_PER_HALLU)
from petridish.nas_control.critic import (
    crawl_data_and_critic_train, ModelSearchInfo)
from petridish.app.options import (
    add_app_arguments, add_model_arguments, add_controller_arguments,
    model_options_processing, options_to_str, is_debug, scale_int_val_with_gpu)
from petridish.nas_control.controller import (
    PetridishRecover, ControllerTypes, QueueSortMethods)
from petridish.app.multi_proc import (
    PetridishServerIPC, mark_stopped, has_stopped, TRAIN_HALLU, TRAIN_MODEL,
    TRAIN_CRITIC_MODEL, TRAIN_CRITIC_HALLU, TRAIN_CRITIC_PARENT,
    NUM_POOLS, stop_mark_fn, mark_failed, is_mark_failure)
from petridish.model.train_eval import (
    train_child, eval_child, get_l_op_order, feature_selection_cutoff)
from petridish.app.local_crawler import launch_local_crawler
from petridish.nas_control.queue_diversity import DiversityOptions
from petridish.analysis.model_util import model_multi_add
from petridish.philly.generator import SCRIPT_TEMPLATE
from petridish.philly.container import (
    local_container_info, get_container_index, get_container_nr_gpu,
    get_runtime_config, get_total_nr_gpu, is_philly)
from petridish.philly.heartbeat import PhillyHeartBeatWorkAround
from petridish.app.directory import (
    previous_trial_log_root, previous_trial_model_root,
    _auto_script_fn, _auto_script_dir,
    _mi_to_dn, _ci_to_dn, _mi_info_save_fn)
from petridish.utils.sample import online_sampling


_fn_from_config_root = os.path.join('petridish', 'app', 'petridish_main.py')

def train_child_remotely(
        model_options, log_dir, child_dir, prev_dir, curr_iter):
    """
    Write a script for crawler to pick up
    """
    auto_scripts_dir = _auto_script_dir(log_dir, is_critic=False)
    script_fn = os.path.join(auto_scripts_dir, _auto_script_fn(curr_iter))
    if not os.path.exists(auto_scripts_dir):
        os.makedirs(auto_scripts_dir)
    options = copy.deepcopy(model_options)
    options.job_type = 'remote_child'
    options.log_dir = log_dir
    options.model_dir = child_dir
    options.prev_model_dir = (
        None if options.child_train_from_scratch else prev_dir)
    options.load = None
    if prev_dir == child_dir and options.init_model_epoch:
        # root models may use extra epochs to have a stable starting point.
        options.max_train_model_epoch = options.init_model_epoch
    # write the auto_script.
    options_str = options_to_str(options)
    script_str = SCRIPT_TEMPLATE.format(
        entry=_fn_from_config_root,
        options=options_str,
        pre_entry_cmds=""
    )
    with open(script_fn, 'wt') as fout:
        fout.write(script_str)
    return log_dir


def train_critic_remotely(controller, data_dir, crawl_dirs,
        log_dir, model_dir, prev_dir, critic_iter, queue_name):
    """
    Write a script for crawler to pick up
    """
    auto_scripts_dir = _auto_script_dir(log_dir, is_critic=True)
    script_fn = os.path.join(
        auto_scripts_dir, _auto_script_fn(critic_iter, queue_name))
    if not os.path.exists(auto_scripts_dir):
        os.makedirs(auto_scripts_dir)
    options = copy.deepcopy(controller.options)
    options.job_type = 'remote_critic'
    options.log_dir = log_dir
    options.model_dir = model_dir
    options.critic_crawl_dirs = crawl_dirs
    options.prev_model_dir = prev_dir
    options.queue_name = queue_name
    options_str = options_to_str(options, ignore=['data_dir'])
    script_str = SCRIPT_TEMPLATE.format(
        entry=_fn_from_config_root,
        options=options_str,
        pre_entry_cmds=""
    )
    with open(script_fn, 'wt') as fout:
        fout.write(script_str)
    return options.log_dir


def fork_and_train_model(ipc, options, log_dir, child_dir, prev_dir,
        model_str, model_iter, parent_iter, search_depth, job_type):
    """
    Spawn a process to write a script for the crawler. then
    wait for the crawler to finish. Aftewards, report to the
    main process.
    """
    entry_func = partial(
        train_child_remotely,
        model_options=options, log_dir=log_dir,
        child_dir=child_dir, prev_dir=prev_dir,
        curr_iter=model_iter)
    #logger.info('Remote child {} will check finish in dir {}'.format(
    #   model_iter, log_dir))
    stop_func = partial(has_stopped, log_dir=log_dir)
    msg_func = lambda model_str=model_str, \
        model_iter=model_iter, parent_iter=parent_iter, \
        search_depth=search_depth, job_type=job_type \
        : [ model_str, model_iter, parent_iter, search_depth, job_type ]
    ipc.spawn(job_type, entry_func, stop_func, msg_func, sleep_time=1)


def fork_and_train_critic(ipc, ctrl, data_dir, crawl_dirs, log_dir,
        model_dir, prev_dir, critic_iter, queue_name, pool):
    """
    Spawn a process to write a script for the crawler. then
    wait for the crawler to finish. Aftewards, report to the
    main process.
    """
    entry_func = partial(
        train_critic_remotely,
        controller=ctrl, data_dir=data_dir,
        crawl_dirs=crawl_dirs, log_dir=log_dir,
        model_dir=model_dir, prev_dir=prev_dir,
        critic_iter=critic_iter, queue_name=queue_name)
    #logger.info('Critic {} will check finish in dir {}'.format(
    #   critic_iter, log_dir))
    stop_func = partial(has_stopped, log_dir=log_dir)
    msg_func = lambda : [ queue_name, critic_iter, pool ]
    ipc.spawn(pool, entry_func, stop_func, msg_func, sleep_time=1)


def server_handle_hallu_message(
        msg_output, controller, mi_info, options, curr_iter):
    """
    Petridish server handles the return message of a forked
    process that watches over a halluciniation job.
    """
    log_dir_root = logger.get_logger_dir()
    q_child = controller.q_child
    model_str, model_iter, _parent_iter, search_depth = msg_output
    # Record performance in the main log
    jr = parse_remote_stop_file(_mi_to_dn(log_dir_root, model_iter))
    if jr is None:
        # job failure: reap the virtual resource and move on.
        logger.info('Failed mi={}'.format(model_iter))
        return curr_iter
    (fp, ve, te, hallu_stats, l_op_indices, l_op_omega) = (
        jr['fp'], jr['ve'], jr['te'], jr['l_stats'],
        jr['l_op_indices'], jr['l_op_omega']
    )
    logger.info(
        ("HALLU : mi={} val_err={} test_err={} "
         "Gflops={} hallu_stats={}").format(
            model_iter, ve, te, fp * 1e-9, hallu_stats))
    mi_info[model_iter].ve = ve
    mi_info[model_iter].fp = fp

    ## compute hallucination related info in net_info
    net_info = net_info_from_str(model_str)
    hallu_locs = net_info.contained_hallucination() # contained
    hallu_indices = net_info.sorted_hallu_indices(hallu_locs)
    # feature selection based on params
    l_fs_ops, l_fs_omega = feature_selection_cutoff(
        l_op_indices, l_op_omega, options)
    separated_hallu_info = net_info.separate_hallu_info_by_cname(
        hallu_locs, hallu_indices, l_fs_ops, l_fs_omega)

    ## Select a subset of hallucination to add to child model
    l_selected = []
    # sort by -cos(grad, hallu) for the indices, 0,1,2,...,n_hallu-1.
    processed_stats = [process_hallu_stats_for_critic_feat([stats]) \
        for stats in hallu_stats]
    logger.info('processed_stats={}'.format(processed_stats))
    logger.info('separated_hallu_info={}'.format(separated_hallu_info))

    # greedy select with gradient boosting
    l_greedy_selected = []
    if options.n_greed_select_per_init:
        greedy_order = sorted(
            range(len(hallu_indices)),
            key=lambda i : - processed_stats[i][0])
        min_select = options.n_hallus_per_select
        max_select = max(min_select, len(hallu_indices) // 2)
        for selected_len in range(min_select, max_select + 1):
            selected = greedy_order[:selected_len]
            l_greedy_selected.append(selected)
        n_greedy_select = len(l_greedy_selected)
        if n_greedy_select > options.n_greed_select_per_init:
            # random choose
            l_greedy_selected = list(np.random.choice(
                l_greedy_selected,
                options.n_greed_select_per_init,
                replace=False))
    # random select a subset
    l_random_selected = []
    if options.n_rand_select_per_init:
        # also try some random samples
        l_random_selected = online_sampling(
            itertools.combinations(
                range(len(hallu_indices)),
                options.n_hallus_per_select
            ),
            options.n_rand_select_per_init)
        np.random.shuffle(l_random_selected)
    l_selected = l_greedy_selected + l_random_selected

    ## for each selected subset of hallu, make a model for q_child
    # since more recent ones tend to be better,
    # we insert in reverse order, so greedy are inserted later.
    for selected in reversed(l_selected):
        # new model description
        child_info = copy.deepcopy(net_info)
        l_hi = [ hallu_indices[s] for s in selected ]
        child_info = child_info.select_hallucination(
            l_hi, separated_hallu_info)
        # Compute initialization stat
        stat = process_hallu_stats_for_critic_feat(
            [hallu_stats[s] for s in selected])
        # update mi_info
        curr_iter += 1
        child_str = child_info.to_str()
        mi_info.append(ModelSearchInfo(
            curr_iter, model_iter, search_depth+1,
            None, None, child_str, stat))
        controller.add_one_to_queue(
            q_child, mi_info, curr_iter, child_info)
    return curr_iter


def server_handle_child_message(
        msg_output, controller, mi_info, options, n_idle, curr_iter):
    """
    Petridish server handles the return message of a forked
    process that watches over a child job.
    """
    log_dir_root = logger.get_logger_dir()
    q_parent, q_hallu = controller.q_parent, controller.q_hallu
    model_str, model_iter, _parent_iter, search_depth = msg_output
    # Record performance in the main log
    jr = parse_remote_stop_file(_mi_to_dn(log_dir_root, model_iter))
    if jr is None:
        # job failure: reap the virtual resource and move on.
        logger.info('Failed mi={}'.format(model_iter))
        return curr_iter
    fp, ve, te = jr['fp'], jr['ve'], jr['te']
    logger.info('CHILD : mi={} val_err={} test_err={} Gflops={}'.format(
        model_iter, ve, te, fp * 1e-9))
    mi_info[model_iter].ve = ve
    mi_info[model_iter].fp = fp

    if (search_depth // 2 < options.max_growth
            and (options.search_max_flops is None
                    or fp < options.search_max_flops)):
        controller.add_one_to_queue(
            q_parent, mi_info, model_iter, None)

    if q_parent.size() > 0:
        # choose a parent.
        pqe = controller.choose_parent(q_parent, mi_info)
        model_str, model_iter, _parent_iter, search_depth = pqe
        logger.info('PARENT : mi={}'.format(model_iter))
        # Create hallucinations on the parent
        net_info_parent = net_info_from_str(model_str)
        n_hallu_per_parent = max(
            1,
            min(controller.n_hallu_per_parent_on_idle, n_idle))
        for _ in range(n_hallu_per_parent):
            net_info = copy.deepcopy(net_info_parent)
            hallus = net_info.sample_hallucinations(
                layer_ops=controller.valid_operations,
                merge_ops=controller.merge_operations,
                prob_at_layer=None,
                min_num_hallus=options.n_hallus_per_init,
                hallu_input_choice=options.hallu_input_choice)
            net_info = net_info.add_hallucinations(
                hallus,
                final_merge_op=controller.hallu_final_merge_op,
                stop_gradient_val=controller.stop_gradient_val,
                hallu_gate_layer=controller.hallu_gate_layer)
            # Update mi_info
            curr_iter += 1
            hallu_str = net_info.to_str()
            mi_info.append(ModelSearchInfo(
                curr_iter, model_iter, search_depth + 1,
                None, None, hallu_str))
            controller.add_one_to_queue(
                q_hallu, mi_info, curr_iter, net_info)
    return curr_iter


def server_handle_critic_message(
        msg_output, controller, mi_info, options):
    """
    Petridish server handles the return message of a forked
    process that watches over a critic job.
    """
    log_dir_root = logger.get_logger_dir()
    model_dir_root = options.model_dir
    queues = controller.queues
    queue_name, new_ci = msg_output
    is_fail, _ = is_mark_failure(
        _ci_to_dn(log_dir_root, new_ci, queue_name))
    if is_fail:
        logger.info('Failed {} ci={}'.format(queue_name, new_ci))
        return
    logger.info('Updating w/ msg of CRITIC {} ci={}'.format(
        queue_name, new_ci))
    # load the new critic
    ctrl_dn = _ci_to_dn(model_dir_root, new_ci, queue_name)
    controller.update_predictor(ctrl_dn, queue_name)
    # as we have new model for critic,
    # remove other old ones if exists.
    ctrl_dns = [_ci_to_dn(model_dir_root, ci, queue_name) \
        for ci in range(new_ci + 1 - controller.n_critic_procs)]
    for ctrl_dn in filter(lambda x : os.path.exists(x), ctrl_dns):
        logger.info('rm -rf {}'.format(ctrl_dn))
        _ = subprocess.check_output(
            'rm -rf {} &'.format(ctrl_dn), shell=True)
    # Sort the affected queue.
    logger.info('Ordering queue {}...'.format(queue_name))
    queue = queues[queue_name]
    controller.update_queue(queue, mi_info)
    logger.info('... done ordering')


def server_init(controller, options):
    """
    Initialize params for server.
    """
    # names and static/fixed info
    log_dir_root = logger.get_logger_dir()
    model_dir_root = options.model_dir

    # Queues.
    queue_names, _queues = controller.init_queues()
    (q_parent, q_hallu, q_child) = (
        controller.q_parent, controller.q_hallu, controller.q_child)
    qname_to_pool = {
        q_child.name : TRAIN_CRITIC_MODEL,
        q_hallu.name : TRAIN_CRITIC_HALLU,
        q_parent.name : TRAIN_CRITIC_PARENT}

    mi_info = []
    if is_debug(options):
        prev_log_root = log_dir_root
        prev_model_root = model_dir_root
    else:
        prev_log_root = previous_trial_log_root(log_dir_root)
        prev_model_root = previous_trial_model_root(model_dir_root)
    is_success = False
    while prev_log_root and prev_model_root:
        logger.info("prev_log_root=\"{}\" && prev_model_root=\"{}\"".format(
            prev_log_root, prev_model_root))
        is_success = controller.recover.recover(
            prev_log_root=prev_log_root,
            log_root=log_dir_root,
            prev_model_root=prev_model_root,
            model_root=model_dir_root,
            q_parent=q_parent,
            q_hallu=q_hallu,
            q_child=q_child,
            mi_info=mi_info)
        if is_success:
            critic_iter = controller.init_predictors(
                prev_log_root, prev_model_root)
            break
        prev_log_root = previous_trial_log_root(prev_log_root)
        prev_model_root = previous_trial_model_root(prev_model_root)
    if not is_success:
        # controller init predictors from scratch
        critic_iter = controller.init_predictors(log_dir_root, model_dir_root)

    if len(mi_info) == 0:
        if options.net_info:
            l_init_net_info = [options.net_info]
            # Need to delete these info because
            # 1. The options is to be used by future children, we want to
            # remove uncessary params.
            # 2. Having both will cause multiple occurance of net_info_str
            # on children scripts, which causes bugs.
            delattr(options, 'net_info')
            options.net_info_str = None
        else:
            l_init_net_info = controller.initial_net_info()
        for mi, net_info in enumerate(l_init_net_info):
            mstr = net_info.to_str()
            # on server model info for each model_iter. Used for critic features.
            # In the order are mi, pi, sd, fp, ve, mstr, stats
            mi_info.append(ModelSearchInfo(mi, mi, 0, None, 2.0, mstr, [0.0]))
            controller.add_one_to_queue(q_child, mi_info, mi, net_info)

    # Job counters
    curr_iter = len(mi_info) - 1

    # queue related counters
    # Model counters and pool resources are reset upon reboot for now.
    n_recv = dict([(qname, 0) for qname in queue_names])
    n_last_train = dict([(qname, 0) for qname in queue_names])
    n_last_mi_save = 0

    # IPC
    pool_sizes = [ 0 ] * NUM_POOLS
    pool_sizes[TRAIN_HALLU] = controller.n_hallu_procs
    pool_sizes[TRAIN_MODEL] = controller.n_model_procs
    for qname in qname_to_pool:
        pool_sizes[qname_to_pool[qname]] = controller.n_critic_procs
    ipc = PetridishServerIPC(pool_sizes, hwm=50)
    ipc.initialize()

    # Server progress workaround
    # This guy wakes up every once a while to increase progress bar a little.
    philly_wa = PhillyHeartBeatWorkAround(max_cnt=options.max_exploration)
    return (
        mi_info,
        ipc,
        qname_to_pool,
        philly_wa,
        curr_iter,
        critic_iter,
        n_recv,
        n_last_train,
        n_last_mi_save
    )


def server_main(
        controller, options,
        hallu_handle=None, child_handle=None, critic_handle=None):
    """
        Server entrance/main.
    """
    model_options_base = options
    log_dir_root = logger.get_logger_dir()
    model_dir_root = options.model_dir
    (
        mi_info,
        ipc,
        qname_to_pool,
        philly_wa,
        curr_iter,
        critic_iter,
        n_recv,
        n_last_train,
        n_last_mi_save
    ) = server_init(controller, options)
    # useful alias:
    (q_hallu, q_child) = (controller.q_hallu, controller.q_child)
    # message handles
    hallu_handle = (
        hallu_handle if hallu_handle else server_handle_hallu_message)
    child_handle = (
        child_handle if child_handle else server_handle_child_message)
    critic_handle = (
        critic_handle if critic_handle else server_handle_critic_message)

    # server main loop
    while ipc.pools.has_active() or q_child.size() > 0 or q_hallu.size() > 0:
        # Launch child/hallu sleepers
        for job_type, queue in zip(
                [TRAIN_HALLU, TRAIN_MODEL], [q_hallu, q_child]):
            # Populate workers util either active is full
            # or option_queue is empty.
            while ipc.pools.has_idle(job_type) and queue.size() > 0:
                model_str, model_iter, parent_iter, search_depth = queue.pop()
                # log the pop order of models. Important for analysis
                logger.info("mi={} pi={} sd={}".format(
                    model_iter, parent_iter, search_depth))
                logger.info("LayerInfoList is :\n{}".format(model_str))
                model_options = copy.deepcopy(model_options_base)
                model_options.net_info = net_info_from_str(model_str)
                fork_and_train_model(ipc=ipc,
                        options=model_options,
                        log_dir=_mi_to_dn(log_dir_root, model_iter),
                        child_dir=_mi_to_dn(model_dir_root, model_iter),
                        prev_dir=_mi_to_dn(model_dir_root, parent_iter),
                        model_str=model_str,
                        model_iter=model_iter,
                        parent_iter=parent_iter,
                        search_depth=search_depth,
                        job_type=job_type)

        # launch critic sleepers
        for qname in [q_child.name, q_hallu.name]:
            _n_new = n_recv[qname] - n_last_train[qname]
            _train_every = controller.controller_train_every
            if _n_new >= _train_every:
                pool = qname_to_pool[qname]
                if ipc.pools.has_idle(pool):
                    n_last_train[qname] = n_recv[qname]
                    ci = critic_iter[qname] = 1 + critic_iter[qname]
                    logger.info('Train critic {} ci={} ...'.format(qname, ci))
                    fork_and_train_critic(
                        ipc=ipc,
                        ctrl=controller,
                        data_dir=options.data_dir,
                        crawl_dirs=log_dir_root,
                        log_dir=_ci_to_dn(log_dir_root, ci, qname),
                        model_dir=_ci_to_dn(model_dir_root, ci, qname),
                        prev_dir=_ci_to_dn(model_dir_root, ci-1, qname),
                        critic_iter=ci,
                        queue_name=qname,
                        pool=pool)
                    logger.info('...Train critic launched')

        logger.info('Listening for message...')
        msg_output, job_type = ipc.get_finished_message()
        if job_type == TRAIN_HALLU:
            n_recv[q_hallu.name] += 1
            curr_iter = hallu_handle(
                msg_output=msg_output,
                controller=controller,
                mi_info=mi_info,
                options=options,
                curr_iter=curr_iter)

        elif job_type == TRAIN_MODEL:
            n_recv[q_child.name] += 1
            n_idle = ipc.pools.num_idle(TRAIN_HALLU)
            curr_iter = child_handle(
                msg_output=msg_output,
                controller=controller,
                mi_info=mi_info,
                options=options,
                n_idle=n_idle,
                curr_iter=curr_iter)

        elif job_type in [
                TRAIN_CRITIC_MODEL, TRAIN_CRITIC_HALLU, TRAIN_CRITIC_PARENT]:
            critic_handle(
                msg_output=msg_output,
                controller=controller,
                mi_info=mi_info,
                options=options)

        ## periodic log/heartbeat/ and exits.
        n_finished = n_recv[q_child.name] + n_recv[q_hallu.name]
        philly_wa.new_heart_beat(cnt=n_finished)
        philly_wa.print_progress_percent()
        # Saving mi_info periodically for training
        # critic, post-processing and recovering.
        np.savez(_mi_info_save_fn(log_dir_root), mi_info=mi_info)
        # we have explore enough models. quit now.
        if n_finished >= options.max_exploration:
            break
    # end while (server main loop)
    logger.info(
        "Exiting server main. n_recv[hallu]={} n_recv[child]={}".format(
            n_recv[q_hallu.name], n_recv[q_child.name]))


def server_exit(log_dir, companion_pids=None):
    """
    At server exit (or interruption) kill companion crawlers
    to release resources.
    """
    mark_stopped(log_dir, is_interrupted=True)
    # kill companion processes.
    if companion_pids is not None:
        for pid in companion_pids.strip().split(','):
            cmd = 'kill -9 {}'.format(pid)
            logger.info('Exiting. killing process {}...'.format(pid))
            subprocess.call(cmd, shell=True)


def parse_remote_stop_file(log_dir):
    """
    Given a stopping file dir (log_dir), will read the stop file and figure out
    whether the run was successful. If succeeded, we return a json object that contains
    the following keys (fp, ve, te, l_stats, l_op_indices, l_op_omega).
    If failed, returns None.
    We assume the l_stats has NOT gone through ``process_hallu_stats_for_critic_feat''.
    """
    is_fail, ret = is_mark_failure(log_dir)
    return None if is_fail else ret


def process_hallu_stats_for_critic_feat(stats):
    """
    Aggregate hallu features into values that can be used to judge hallus.
    """
    ret = [0.0 for _ in range(NUM_STATS_PER_HALLU)]
    for stat in stats:
        for i, s in enumerate(stat):
            ret[i] += s
    dotprod, g_l2, h_l2 = ret
    if g_l2 == 0.0 or h_l2 == 0.0:
        return [ 0.0 ]
    return [dotprod / np.sqrt(g_l2 * h_l2)]


def server_handle_child_message_soft_vs_hard(
        msg_output, controller, mi_info, options, n_idle, curr_iter):
    """
    Special replacement of server_handle_child_message for
    experimenting on soft init vs. hard init.

    This is for experiment only.
    TODO reuse code with regular server_handle_child_message?
    """
    log_dir_root = logger.get_logger_dir()
    q_parent, q_hallu = controller.q_parent, controller.q_hallu
    model_str, model_iter, _parent_iter, search_depth = msg_output
    # Record performance in the main log
    jr = parse_remote_stop_file(_mi_to_dn(log_dir_root, model_iter))
    if jr is None:
        # job failure: reap the virtual resource and move on.
        logger.info('Failed mi={}'.format(model_iter))
        return curr_iter
    fp, ve, te = jr['fp'], jr['ve'], jr['te']
    logger.info('CHILD : mi={} val_err={} test_err={} Gflops={}'.format(
        model_iter, ve, te, fp * 1e-9))
    mi_info[model_iter].ve = ve
    mi_info[model_iter].fp = fp

    if search_depth > 0:
        return curr_iter

    controller.n_hallu_per_parent_on_idle = 1
    # for soft vs hard experiment, only root generates hallu.
    controller.add_one_to_queue(q_parent, mi_info, model_iter, None)
    if q_parent.size() > 0:
        # choose a parent.
        pqe = controller.choose_parent(q_parent, mi_info)
        model_str, model_iter, _parent_iter, search_depth = pqe
        logger.info('PARENT : mi={}'.format(model_iter))
        # Create hallucinations on the parent
        net_info_parent = net_info_from_str(model_str)

        # this experiment only creates one hallu from the root
        hallus = net_info_parent.sample_hallucinations(
            layer_ops=controller.valid_operations,
            merge_ops=controller.merge_operations,
            prob_at_layer=None,
            min_num_hallus=options.n_hallus_per_init,
            hallu_input_choice=options.hallu_input_choice)

        for netmorph_method in ['hard', 'soft']:
            controller.set_netmorph_method(netmorph_method)
            net_info = copy.deepcopy(net_info_parent)
            net_info = net_info.add_hallucinations(
                hallus,
                final_merge_op=controller.hallu_final_merge_op,
                stop_gradient_val=controller.stop_gradient_val,
                hallu_gate_layer=controller.hallu_gate_layer)
            # Update mi_info
            curr_iter += 1
            hallu_str = net_info.to_str()
            mi_info.append(ModelSearchInfo(
                curr_iter, model_iter, search_depth + 1,
                None, None, hallu_str))
            controller.add_one_to_queue(
                q_hallu, mi_info, curr_iter, net_info)
    return curr_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_app_arguments(parser)
    add_model_arguments(parser)
    add_controller_arguments(parser)
    DiversityOptions.add_parser_arguments(parser)
    PetridishRecover.add_parser_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = model_options_processing(args)

    if args.job_type == 'main':
        # The log is set to be {args.log_dir}/petridish_main/log.log
        # FIXME: A bit weird that some of the utils are in ANN repo.
        # Might be good to refactor.
        ann_app_utils.log_init(args, None)
        logger.info("App has the following unknown arguments : {}".format(unknown))
        log_dir = logger.get_logger_dir()

        # Update nr_gpu related param based on rutime config and all containers.
        # FIXME: This block of code below is Philly specific. But it is okay because
        # if we are not running on Philly then the functions below will check for Philly
        # environment and then skip automatically if it is not Philly.
        runtime_config = get_runtime_config()
        args.total_nr_gpu = get_total_nr_gpu(config=runtime_config)
        logger.info("Main has access to {} gpus".format(args.total_nr_gpu))
        # Update more nr_gpu related param based on local container.
        cinfo = local_container_info(config=runtime_config)
        container_id = get_container_index(cinfo)
        logger.info("Container index = {}".format(container_id))
        if cinfo is not None:
            nr_gpu = get_container_nr_gpu(cinfo)
            args.nr_gpu = nr_gpu
        logger.info("Container nr_gpu = {}".format(args.nr_gpu))

        # Automatically set up the log directory
        auto_dir = _auto_script_dir(log_dir, is_critic=False, is_log_dir_root=True)
        if not os.path.exists(auto_dir):
            try:
                os.makedirs(auto_dir)
            except Exception as e:
                # multi-container could have multiple make_dir and race.
                if not os.path.exists(auto_dir):
                    raise e

        # Launching local crawler if we opt to do so.
        if args.launch_local_crawler:
            postfix = "" if container_id is None else str(container_id)
            crawler_log_dir = os.path.join(log_dir, 'launcher' + postfix)
            if not os.path.exists(crawler_log_dir):
                os.makedirs(crawler_log_dir)

            # Number of maximum jobs on this crawler is based on local container.
            n_parallel = scale_int_val_with_gpu(
                (args.n_hallu_procs_per_gpu + args.n_model_procs_per_gpu +
                  args.n_critic_procs_per_gpu),
                args.nr_gpu
            )
            crawler = launch_local_crawler(
                auto_dir, args.nr_gpu, crawler_log_dir,
                n_parallel, args.num_init_use_all_gpu)
            if not args.companion_pids:
                args.companion_pids = str(crawler.pid)
            else:
                args.companion_pids = args.companion_pids + ',' + str(crawler.pid)
        # Kill local crawler at exit.
        logger.info("KILL this pids after main dies : {}".format(args.companion_pids))
        atexit.register(server_exit, log_dir, args.companion_pids)

        if (args.launch_local_crawler and
                container_id is not None and int(container_id) > 0):
            # If this is not the chief container,
            # then we wait for the launcher.
            crawler.join()
        else:
            hallu_handle, child_handle, critic_handle = None, None, None
            # Server.
            ctrl_cls = ControllerTypes.type_idx_to_controller_cls(args.controller_type)
            controller = ctrl_cls(args)
            if args.grow_petridish_version == 'soft_vs_hard':
                # set some constant params.
                args.max_exploration = 5
                args.max_growth = 1
                child_handle = server_handle_child_message_soft_vs_hard

            elif args.grow_petridish_version == 'inc_vs_scratch':
                #grow_petridish_inc_vs_scratch(controller, args, model_cls)
                raise NotImplementedError("Implement using mp + args.child_train_from_scratch")

            elif args.grow_petridish_version == 'mp':
                # will use all default handles
                pass

            server_main(
                controller=controller,
                options=args,
                hallu_handle=hallu_handle,
                child_handle=child_handle,
                critic_handle=critic_handle)
            mark_stopped(log_dir)

    elif args.job_type == 'remote_child':
        # If there is a race where multiple child/critic are running on the same job
        # then there is a chance that an error may be raised due to
        # non-thread-safe code.
        # This is a GOOD, since it kills redundant jobs.
        # The resource limit of server is also not wasted, because the pool is based on number of
        # scripts generated by the server and is not based on the number of remote jobs running.
        # The resource limit of on-philly crawler/launcher is not wasted, because the aborted script
        # will be removed.
        #
        # We don't catch this error, as we want to kill the job any way.
        # We also do not want to catch other errors by accident.

        model_cls = ControllerTypes.type_idx_to_child_model_cls(
            args.controller_type)
        logger.set_logger_dir(args.log_dir, action='d')
        net_info = args.net_info
        try:
            train_child(model_cls, args, args.log_dir,
                args.model_dir, args.prev_model_dir)
            tf.reset_default_graph()

            # grep from model file for feature selection
            l_op_indices, l_op_omega = get_l_op_order(net_info, args.model_dir)
            # eval child for ve and stats
            val_ret = eval_child(model_cls, args,
                args.log_dir, args.model_dir, collect_hallu_stats=True)
            ve = val_ret[0]
            stats = val_ret[1:]
            l_stats = [ val_ret[idx:idx+NUM_STATS_PER_HALLU] \
                for idx in range(1, len(val_ret), NUM_STATS_PER_HALLU) ]
            # grep eval file for computation
            multi_add = model_multi_add(os.path.join(args.log_dir, 'log.log'))
            fp = multi_add * 2.0
            # test the model
            if args.do_validation:
                args.do_validation = False
                args.compute_hallu_stats = False
                test_ret = eval_child(model_cls, args,
                    args.log_dir, args.model_dir, collect_hallu_stats=False)
                te = test_ret[0]
            else:
                te = ve
            # form stopping message for main.
            json_ret = dict()
            json_ret['ve'] = ve
            json_ret['te'] = te
            json_ret['fp'] = fp
            json_ret['l_stats'] = l_stats
            json_ret['l_op_indices'] = l_op_indices
            json_ret['l_op_omega'] = l_op_omega
            ret_str = dumps(json_ret)
            msg_func = lambda : ret_str
            mark_stopped(args.log_dir, msg_func=msg_func)
            # Go to parse_remote_stop_file for how this msg is parsed.
        except Exception as e:
            mi = os.path.basename(os.path.normpath(args.model_dir))
            logger.info("mi={} failed: {}".format(mi, e))
            # TODO differentiate OOM and Unknown:
            # tensorflow.python.framework.errors_impl.UnknownError
            # tensorflow.python.framework.errors_impl.ResourceExhaustedError
            mark_failed(args.log_dir)
            traceback.print_exc()
            raise
        # pack the info for the stop file, see parse_remote_stop_file forr unpacking


    elif args.job_type == 'remote_critic':
        logger.set_logger_dir(args.log_dir, action='d')
        ctrl_cls = ControllerTypes.type_idx_to_controller_cls(args.controller_type)
        controller = ctrl_cls(args)
        assert args.queue_name is not None, 'remote critic must know which queue to judge'
        try:
            crawl_data_and_critic_train(controller=controller,
                data_dir=os.path.join(args.data_dir, 'petridish'),
                crawl_dirs=args.critic_crawl_dirs,
                log_dir=args.log_dir,
                model_dir=args.model_dir,
                prev_dir=args.prev_model_dir,
                vs_name=args.queue_name,
                store_data=args.store_critic_data)
            mark_stopped(args.log_dir)
        except Exception as e:
            ci = os.path.basename(os.path.normpath(args.model_dir))
            logger.info("qname={} ci={} failed: {}".format(args.queue_name, ci, e))
            mark_failed(args.log_dir)
            traceback.print_exc()
            raise
