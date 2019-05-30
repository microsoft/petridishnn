import os
import json


def is_philly():
    # DEBUG for PT
    # return False
    config_fn = os.environ.get('PHILLY_RUNTIME_CONFIG', None)
    return config_fn is not None and len(config_fn) > 0

"""
Runtime config related
"""
def get_runtime_config():
    # Note: Philly always has this environment variable
    # defined. So we can check if we are running in Philly.
    # The value of this variable is a json file which contains
    # all kinds of Philly specific information. Echo had to figure
    # this out himself and this was not documented well on the Philly
    # website.
    fn = os.environ.get('PHILLY_RUNTIME_CONFIG', None)
    if not fn:
        return None
    with open(fn, 'rb') as fin:
        config = json.load(fin)
    return config

def get_local_ip():
    ssh_conn = os.environ.get('SSH_CONNECTION', None)
    if not ssh_conn:
        return None
    ssh_conn = ssh_conn.strip().split()
    local_ip = ssh_conn[2]
    local_port = int(ssh_conn[3])
    return local_ip, local_port

def local_container_info(config=None):
    # The config found from Philly has config 
    # for every container for this job. This function 
    # finds info for us. Echo? 
    ip_port = get_local_ip()
    if ip_port is None:
        return None
    local_ip, local_port = get_local_ip()

    if config is None:
        config = get_runtime_config()

    if not config or not local_ip:
        return None

    for cname in config['containers']:
        info = config['containers'][cname]
        if info['ip'] == local_ip and info['sshdPort'] == local_port:
            return info
    return None

def get_container_index(info=None):
    if info is None:
        info = local_container_info()
    if info is None:
        return None
    return info['index']

def get_container_nr_gpu(info=None):
    if info is None:
        info = local_container_info()
    if info is None:
        return None
    return info['containerGpuCount']


def get_total_nr_gpu(config=None):
    if config is None:
        config = get_runtime_config()
    if config is None:
        return None
    nr_gpu = 0
    for cname in config['containers']:
        info = config['containers'][cname]
        nr_gpu += get_container_nr_gpu(info)
    return nr_gpu
