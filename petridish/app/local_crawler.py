import os
import re
import subprocess
import time
import multiprocessing as mp
import shutil
import numpy as np
import errno

from tensorpack.utils import logger

def crawl_local_auto_scripts_and_launch(
        auto_dir, nr_gpu=1, launcher="", n_parallel=10000,
        num_init_use_all_gpu=2):
    """
    Process overview: if there is available resource limit (see n_parallel below), we
    launch a job. We first copy the job script into our local log dir to prevent other
    process from also launchng it. We remove the job script from the original auto_dir after
    copying. If either "copy" or "remove" fails, it means other process already owned the job
    and we abort by removing the job from our log dir.

    If we have the job script in our log dir, we actually launch it by translating
    its argument into a cmd "( ... && python xxxx ; rm -rf xxx.sh) &". We call this command
    to do this job in a different process. The cmd also includes a "...; rm xx.sh" to release
    the resource counted by n_parallel, regardless of the status of the remote job.

    Args
    auto_dir (str) : where to look for auto_scripts
    nr_gpu (int) : the numebr of gpu the crawler can use. A round robin schedule is used.
    launcher (str) : the name of the launcher, which is used for logger.
    n_parallel (int) : max number of parallel jobs. We count the number of n_parallel
        using the number of .sh files in the launcher's own log dir. Hence it is IMPERATIVE
        for each launched job to remove its own .sh after finishing regardless of sucess/failure.
        Bugged xx.sh are copied into xx.sh.fixme to avoid resource leak.
    num_init_use_all_gpu (int) : if mi < num_init_use_all_gpu then it will use
        all availabel gpu. This is for the initial jobs to be faster
    """
    device = -1
    while True:
        time.sleep(1)
        if os.path.exists(auto_dir):
            break
    logger.info("Found the auto_dir {}".format(auto_dir))
    launch_log = logger.get_logger_dir()

    # python 2 vs 3 crap
    check_errno = False
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = OSError
        check_errno = True
    logger.info("Crawler check_errno = {}".format(check_errno))

    def _newFileNotFound():
        e = FileNotFoundError()
        e.errno = errno.ENOENT
        return e

    def _isFileNotFound(e):
        if hasattr(e, 'errno') and e.errno is not None and e.errno == errno.ENOENT:
            return True
        if check_errno:
            return False
        return isinstance(e, FileNotFoundError)

    while True:
        time.sleep(np.random.uniform(low=1.0, high=5.0))
        n_running = len(list(filter(lambda x : x.endswith('.sh'), os.listdir(launch_log))))
        if n_running >= n_parallel:
            continue

        l_scripts = os.listdir(auto_dir)
        np.random.shuffle(l_scripts)
        for script in l_scripts:
            if script.endswith('.lck'):
                # this is a lock file. ignore
                continue
            auto_script = os.path.join(auto_dir, script)
            auto_script_tmp = os.path.join(launch_log, script)
            lock = auto_script + '.lck'
            if os.path.exists(lock):
                # someone early has locked the file. ignore
                continue
            if not os.path.exists(auto_script):
                # someone early has removed the script. ignore
                continue

            try:
                with open(lock, 'wt'):
                    shutil.copyfile(auto_script, auto_script_tmp)
                    if not os.path.exists(auto_script):
                        # this is important. It makes sure that pycmd is valid.
                        # Remove the tmp, if we found that we are not the first.
                        os.remove(auto_script_tmp)
                        raise _newFileNotFound()
                    # this may raise error due to race.
                    # All process could raise here due to strange iteractions.
                    os.remove(auto_script)
            except Exception as e:
                if _isFileNotFound(e):
                    # this means someone else removed the auto_script
                    # before we did, so that guy is to launch
                    if os.path.exists(auto_script_tmp):
                        os.remove(auto_script_tmp)
                    logger.info("Race on script {}".format(script))
                else:
                    logger.info("Crazy Race on {} : {} : {}".format(script, e.__class__, e))
                    # Other errors means race within os.remove ...
                    # which means maybe none of them succeed to remove ...
                    # so all launch.

            while os.path.exists(lock):
                # every process that opened the lock should attempt to remove it.
                try:
                    os.remove(lock)
                    break
                except:
                    logger.info("Race on rm lock of {}".format(script))

            # this file is only accessible by the current launcher. No need to lock.
            if os.path.exists(auto_script_tmp):
                # Translate
                pycmd, n_job_gpu = script_to_local_cmd(
                    auto_script_tmp, nr_crawler_gpu=nr_gpu,
                    num_init_use_all_gpu=num_init_use_all_gpu)
                if pycmd is None:
                    logger.info("FIXME: {} failed on {}".format(launcher, script))
                    # rename so that it is no longer in the resource limit.
                    os.rename(auto_script_tmp, auto_script_tmp+'.fixme')
                    continue
                visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                if visible_gpus:
                    visible_gpus = visible_gpus.strip().split(',')
                    assert len(visible_gpus) >= nr_gpu, \
                        '{} != {}'.format(len(visible_gpus), nr_gpu)
                else:
                    visible_gpus = [str(gpu_id) for gpu_id in range(nr_gpu)]
                job_device = []
                for _ in range(n_job_gpu):
                    device = (device + 1) % nr_gpu
                    job_device.append(visible_gpus[device])
                job_device = ','.join(job_device)
                cmd = '(export CUDA_VISIBLE_DEVICES="{device}" && {pycmd} >> {out_fn} ; rm -rf {script}) &'.format(\
                    device=job_device, pycmd=pycmd,
                    out_fn=os.path.join(launch_log, 'remote_stdout.txt'),
                    script=auto_script_tmp)
                logger.info("Launch job {} on GPU {} by {}".format(\
                    script, job_device, launcher))
                # launch the script in a different process
                subprocess.call(cmd, shell=True)


def script_to_local_cmd(
        script, data_dir=None, log_dir=None,
        model_dir=None, nr_crawler_gpu=None,
        num_init_use_all_gpu=2):
    basename = os.path.basename(os.path.normpath(script))
    keep_nr_gpu = False
    try:
        mi = int(basename.split('.')[0])
        # TODO in the future makes thie determiend by the crawler.
        # The crawler will determine this based on how busy it is,
        # or how many jobs it found.
        # TODO Crawler also needs info for cases where
        # models are too large for 1 GPU.
        if mi < num_init_use_all_gpu:
            logger.info("Will use all gpu for mi={}".format(mi))
            keep_nr_gpu = True
    except:
        pass

    with open(script, 'rt') as fin:
        is_cmd = False
        cmd = []
        dn_patterns = [r'--(data_dir)=(.*) ', r'--(log_dir)=(.*) ', r'--(model_dir)=(.*) ']
        dns = dict()
        _batch_size = None
        nr_gpu = None
        for line in fin:
            line = line.strip()
            reret = re.search(r'python.*CONFIG_DIR.*py \\$', line)
            if reret is not None:
                #entry = re.search(r'([A-Za-z0-9_]*.py)', line).group(0)
                is_cmd = True
            if is_cmd:
                if len(line) == 0:
                    break
                reret = re.search(r'-+batch_size=([0-9]*) ', line)
                if reret is not None:
                    batch_size = int(reret.group(1))
                    line = re.sub(reret.group(0), '', line)
                reret = re.search(r'-+nr_gpu=([0-9]*) ', line)
                if reret is not None:
                    nr_gpu = int(reret.group(1))
                    line = re.sub(reret.group(0), '', line)
                reret = re.search(r'-+init_lr=([0-9\.]*) ', line)
                if reret is not None:
                    _init_lr = float(reret.group(1))
                    line = re.sub(reret.group(0), '', line)
                for pattern in dn_patterns:
                    reret = re.search(pattern, line)
                    if reret:
                        dns[reret.group(1)] = reret.group(2)
                        line = re.sub(reret.group(0), '', line)
                line = line.strip()
                if len(line) > 0 and line != '\\':
                    if line[-1] != '\\':
                        cmd.append(line + '\\')
                        break
                    else:
                        cmd.append(line)
        nr_gpu = nr_crawler_gpu if keep_nr_gpu else 1
        cmd.append('--nr_gpu={} \\'.format(nr_gpu))
        data_dir = data_dir if data_dir else dns['data_dir']
        cmd.append('--data_dir={} \\'.format(data_dir))
        log_dir = log_dir if log_dir else dns['log_dir']
        cmd.append('--log_dir={} \\'.format(log_dir))
        model_dir = model_dir if model_dir else dns['model_dir']
        cmd.append('--model_dir={} \\'.format(model_dir))
        return '\n'.join(cmd), nr_gpu


def local_crawler_main(
        auto_dir, nr_gpu, launch_log_dir,
        n_parallel=10000, num_init_use_all_gpu=2):
    """
    Args:
    auto_dir (str) : dir for looking for xxx.sh to run
    nr_gpu (int): Number of gpu on local contaienr
    launch_log_dir (str) : where the launcher logs stuff and hold tmp scripts.
    n_parallel (int) : maximum number of parallel jobs.
    num_init_use_all_gpu (int) : num of init jobs that will use all gpu
    """
    logger.set_logger_dir(launch_log_dir, action='d')
    launcher = os.path.basename(os.path.normpath(launch_log_dir))
    crawl_local_auto_scripts_and_launch(
        auto_dir, nr_gpu, launcher, n_parallel, num_init_use_all_gpu)

def launch_local_crawler(*args):
    p = mp.Process(target=local_crawler_main, args=args)
    p.start()
    return p