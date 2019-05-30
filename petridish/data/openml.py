# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import scipy.io.arff as arff
import bisect
import json
import os, sys
import subprocess
import tensorflow as tf

from tensorpack.dataflow import RNGDataFlow, BatchData, PrefetchData
from tensorpack.callbacks import Inferencer

from tensorpack.dataflow import DataFlow, PrefetchDataZMQ, \
    PrefetchData, \
    MapDataComponent, AugmentImageComponent, BatchData
from tensorpack.dataflow import imgaug
from tensorpack.utils import logger


def maybe_download_dataset(dataset_idx, json_dir=None, data_dir=None,
        force_download=False, disable_download=True):
    json_fn = os.path.join(json_dir, str(dataset_idx) + '.json')
    data_fn = os.path.join(data_dir, str(dataset_idx) + '.arff')
    if os.path.exists(json_fn) and not force_download:
        print("Json info and data already exists.")
    else:
        if disable_download:
            raise ValueError("{} should exist but not".format(json_fn))
        import wget, glob
        url = "https://www.openml.org/d/{dataset_idx}/json".format(dataset_idx=dataset_idx)
        print("Downloading JSON file from url {}".format(url))
        json_fn = wget.download(url, json_fn)
        fns = glob.glob('{}*tmp'.format(json_fn))
        for fn in fns:
            cmd = 'rm {}'.format(fn)
            print("remove tmp file with cmd : {}".format(cmd))
            subprocess.call(cmd, shell=True)

    with open(json_fn, 'rt') as json_in:
        lines = []
        for line in json_in:
            lines.append(line.strip())
    ss = ''.join(lines)
    data_info = json.loads(ss)

    #target_attr = data_info.get('default_target_attribute', None)
    target_attr = None
    if target_attr is None:
        n_targets = 0
        for feat_info in data_info['features']:
            if int(feat_info.get('target', 0)) > 0:
                target_attr = feat_info['name']
                n_targets += 1
        if n_targets != 1:
            raise Exception("current logic only support 1d prediction at dataset_idx {}".format(dataset_idx))

    if os.path.exists(data_fn) and not force_download:
        print("data arff already exists")
    else:
        if disable_download:
            raise ValueError("{} should exist but not".format(data_fn))
        import wget
        import glob
        # dataset url
        url = data_info['url']
        print("Downloading dataset {} from url {}".format(dataset_idx, url))
        data_fn = wget.download(url, out=data_fn)
        fns = glob.glob('{}*tmp'.format(data_fn))
        for fn in fns:
            cmd = 'rm {}'.format(fn)
            print("remove tmp file with cmd : {}".format(cmd))
            subprocess.call(cmd, shell=True)

    return data_fn, target_attr



def get_arff_data(fn, target_attr='class', check_size_only=False):
    file_stat = os.stat(fn)
    if check_size_only:
        print("{} has size {}MB".format(fn, file_stat.st_size * 1e-6))
        return None
    data, meta = arff.loadarff(fn)

    if not target_attr in meta.names():
        raise Exception("Dataset {} is broken: target_attr {} not in meta".format(fn, target_attr))

    # problem type regression/classification
    if meta[target_attr][0] == 'numeric':
        num_classes = 0
        pred_type = tf.float32
    else:
        num_classes = len(meta[target_attr][1])
        pred_type = tf.int32
        pred_val2idx = dict()
        for vi, val in enumerate(meta[target_attr][1]):
            pred_val2idx[val] = vi

    # feature names, types and ranges
    feat_names = list(filter(lambda x : x != target_attr, meta.names()))
    n_feats = len(feat_names)
    feat_types = [tf.float32 for _ in range(n_feats)]
    feat_dims = [None for _ in range(n_feats)]
    feat_val2idx = [None for _ in range(n_feats)]
    for i, name in enumerate(feat_names):
        if meta[name][0] == 'numeric':
            continue
        feat_types[i] = tf.int32
        feat_dims[i] = len(meta[name][1])
        feat_val2idx[i] = dict()
        for vi, val in enumerate(meta[name][1]):
            feat_val2idx[i][val] = vi

    n_data = len(data)
    dps = [[None] * n_data for _ in range(n_feats + 1) ]
    feat_means = [ 0. for _ in range(n_feats)]
    feat_vars = [ 0. for _ in range(n_feats)]
    for xi, x in enumerate(data):
        for di, dname in enumerate(feat_names):
            val = x[dname]
            if feat_types[di] == tf.float32:
                val = float(val)
                dps[di][xi] = val
                feat_means[di] += val
                feat_vars[di] += val * val
            else:
                val = val.decode("utf-8")
                dps[di][xi] = int(feat_val2idx[di][val])

        if num_classes == 0:
            dps[-1][xi] = float(x[target_attr])
        else:
            val = x[target_attr].decode("utf-8")
            dps[-1][xi] = int(pred_val2idx[val])

    feat_types.append(pred_type)
    feat_dims.append(None)
    feat_means = [ z / float(n_data) for z in feat_means ]
    feat_stds = [ np.sqrt((sq / float(n_data) - m * m)) for sq, m in zip(feat_vars, feat_means)]
    return dps, feat_types, feat_dims, n_data, num_classes, feat_means, feat_stds


class LoadedArffDataFlow(RNGDataFlow):

    def __init__(self, dps_ys, split, shuffle=True,  do_validation=False):
        super(LoadedArffDataFlow, self).__init__()
        self.shuffle = shuffle
        self.dps = dps_ys # this should be a list of n x d_i mat, the last one is pred ys
        n_samples = len(dps_ys[-1])
        self.init_indices = list(range(n_samples))
        np.random.seed(180451613)
        np.random.shuffle(self.init_indices)

        if split == 'all':
            self._offset = 0
            self._size = n_samples
        elif split == 'train':
            self._offset = 0
            if do_validation:
                self._size = n_samples * 8 // 10
            else:
                self._size = n_samples * 9 // 10
        elif split == 'val' or split == 'validation':
            if do_validation:
                self._offset = n_samples * 8 // 10
                self._size = n_samples * 9 // 10 - self._offset
            else:
                self._offset = n_samples * 9 // 10
                self._size = n_samples - self._offset
        elif do_validation and split == 'test':
            self._offset = n_samples * 9 // 10
            self._size = n_samples - self._offset

    def size(self):
        return self._size

    def get_data(self):
        idxs = [ i for i in self.init_indices[self._offset:(self._offset + self._size)]]
        if self.shuffle:
            np.random.shuffle(idxs)
        for k in idxs:
            yield [dp[k] for dp in self.dps]

def get_dataset_by_id(idx, data_dir_root, check_size_only=False, disable_download=True):
    data_dir = os.path.join(data_dir_root, 'openml')
    json_dir = os.path.join(data_dir_root, 'openml', 'json_dir')
    fn, target_attr = maybe_download_dataset(idx, json_dir=json_dir, data_dir=data_dir,
        disable_download=disable_download)
    return get_arff_data(fn, target_attr, check_size_only)

def get_openml_dataflow(idx, data_root, splits=[], do_validation=False):
    (dps_ys, types, dims, n_data,
    num_classes, feat_means, feat_stds) = get_dataset_by_id(idx, data_root)
    l_ds = dict()
    for split in splits:
        l_ds[split] =  LoadedArffDataFlow(
            dps_ys, split, shuffle=True, do_validation=do_validation)
    return l_ds, types, dims, n_data, num_classes, feat_means, feat_stds

# copy paste from the paper: https://arxiv.org/pdf/1802.04064.pdf
cbb_openml_indices = [
    3, 6, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 23, 26, 28, 30, 31, 32,
    36, 37, 39, 40, 41, 43, 44, 46, 48, 50, 53, 54, 59, 60, 61, 62, 150,
    151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 180, 181, 182, 183,
    184, 187, 189, 197, 209, 223, 227, 273, 275, 276, 277, 278, 279, 285, 287,
    292, 293, 294, 298, 300, 307, 310, 312, 313, 329, 333, 334, 335, 336, 337,
    338, 339, 343, 346, 351, 354, 357, 375, 377, 383, 384, 385, 386, 387, 388,
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 444, 446,
    448, 450, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 472, 475,
    476, 477, 478, 479, 480, 554, 679, 682, 683, 685, 694, 713, 714, 715, 716,
    717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731,
    732, 733, 734, 735, 736, 737, 740, 741, 742, 743, 744, 745, 746, 747, 748,
    749, 750, 751, 752, 753, 754, 755, 756, 758, 759, 761, 762, 763, 764, 765,
    766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780,
    782, 783, 784, 785, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797,
    799, 800, 801, 803, 804, 805, 806, 807, 808, 811, 812, 813, 814, 815, 816,
    817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 832,
    833, 834, 835, 836, 837, 838, 841, 843, 845, 846, 847, 848, 849, 850, 851,
    853, 855, 857, 859, 860, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871,
    872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 884, 885, 886, 888,
    891, 892, 893, 894, 895, 896, 900, 901, 902, 903, 904, 905, 906, 907, 908,
    909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923,
    924, 925, 926, 927, 928, 929, 931, 932, 933, 934, 935, 936, 937, 938, 941,
    942, 943, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 958,
    959, 962, 964, 965, 969, 970, 971, 973, 974, 976, 977, 978, 979, 980, 983,
    987, 988, 991, 994, 995, 996, 997, 1004, 1005, 1006, 1009, 1011, 1012, 1013,
    1014, 1015, 1016, 1019, 1020, 1021, 1022, 1025, 1026, 1036, 1038, 1040,
    1041, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1054, 1055, 1056, 1059,
    1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073,
    1075, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087,
    1088, 1100, 1104, 1106, 1107, 1110, 1113, 1115, 1116, 1117, 1120, 1121,
    1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133,
    1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146,
    1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158,
    1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1169, 1216, 1217, 1218,
    1233, 1235, 1236, 1237, 1238, 1241, 1242, 1412, 1413, 1441, 1442, 1443,
    1444, 1449, 1451, 1453, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1470,
    1471, 1472, 1473, 1475, 1481, 1482, 1483, 1486, 1487, 1488, 1489, 1496, 1498
]

# 21 could not convert sparse str to float; 6 cannot convert sparse nominal of 1/-1
# 2 could not find nominal field ; 1 exception due to target_attr not found
# 2 StopIteration
cbb_openml_indices_failed = [
    189, 273, 292, 293, 310, 351, 354, 357, 383, 384, 385, 386, 387, 388, 389,
    390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 1048, 1073,
    1100, 1169, 1241, 1242
]

def len_cbb_indices():
    return len(cbb_openml_indices)

if __name__ == '__main__':

    from urllib.request import HTTPError, ContentTooShortError
    import time
    failed_indices = []
    data_dir = '/data/data'
    try:
        os.makedirs(os.path.join(data_dir, 'openml', 'json_dir'))
    except:
        pass

    for cnt, i in enumerate(cbb_openml_indices):
        print("{}/{} : urlid={} ... ".format(cnt+1, len(cbb_openml_indices), i))
        start = time.time()
        try:
            ret = get_dataset_by_id(i, data_dir, check_size_only=True, disable_download=False)
        except (HTTPError, ContentTooShortError) as e:
            print("\n wget failed on {} with error {}".format(i, e))
            failed_indices.append(i)
        print("...done {} sec".format(time.time() - start))
    print("The indices that failed: {}".format(failed_indices))

    #fn = maybe_download_dataset(31)
    #ret = get_arff_data(fn)
