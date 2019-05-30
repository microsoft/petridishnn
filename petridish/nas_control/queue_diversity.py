# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import heapq

def search_tree_dist(mi1, mi2, mi_info):
    info1 = mi_info[mi1]
    info2 = mi_info[mi2]

    if info1.sd < info2.sd:
        tmp = info1
        info1 = info2
        info2 = tmp

    def _up_k(info, k=None):
        n_edges = 0
        while info.pi != info.mi and (k is None or n_edges < k):
            info = mi_info[info.pi]
            n_edges += 1
        return info, n_edges

    k = info1.sd - info2.sd
    info1, n_edges = _up_k(info1, k)
    dist = n_edges

    while info1.mi != info2.mi and (info1.mi != info1.pi or info2.mi != info2.pi):
        if info1.pi != info1.mi:
            info1 = mi_info[info1.pi]
            dist += 1
        if info2.pi != info2.mi:
            info2 = mi_info[info2.pi]
            dist += 1
    if info1.mi != info2.mi:
        # this implies that there are multiple roots and the two info are from
        # different roots
        # TODO make this an option or precomputed.
        dist += 5
    return dist


def search_depth_diff(mi1, mi2, mi_info):
    info1 = mi_info[mi1]
    info2 = mi_info[mi2]
    return float(np.abs(info1.sd - info2.sd))


def prediction_diff(mi1, mi2, mi_info):
    info1 = mi_info[mi1]
    info2 = mi_info[mi2]
    # Go to each guy's model_dir and look for prediction vec /wrong vec on validation set
    # then compare the diff.

    # Require changes of eval_child to store such wrong_vec
    return 0.0


def flops_diff(mi1, mi2, mi_info):
    info1 = mi_info[mi1]
    info2 = mi_info[mi2]
    if info1.fp is None or info2.fp is None:
        return 0.0
    return np.abs(info1.fp - info2.fp)


def compute_diversity(l_diff, div_opts):
    if div_opts.combine_type == 'avg':
        diff = np.mean(l_diff)
    elif div_opts.combine_type == 'max':
        # max dist
        diff = np.max(l_diff)
    elif div_opts.combine_type == 'min':
        # min dist 
        diff = np.min(l_diff)
    # The lower the diff, the more similar the sample is, the lower the diversity score.
    return diff


class DiversityOptions(object):

    def __init__(self, options):
        self.max_frac = options.diversity_max_frac
        self.top_k = options.diversity_top_k
        self.prediction_diff_w = options.diversity_prediction_diff_w
        self.search_depth_diff_w = options.diversity_search_depth_diff_w
        self.search_tree_dist_w = options.diversity_search_tree_dist_w
        self.combine_type = options.diversity_combine_type
        self.flops_diff_w = options.diversity_flops_diff_w
        
        delattr(options, 'diversity_max_frac')
        delattr(options, 'diversity_top_k')
        delattr(options, 'diversity_prediction_diff_w')
        delattr(options, 'diversity_search_depth_diff_w')
        delattr(options, 'diversity_search_tree_dist_w')
        delattr(options, 'diversity_combine_type')
        delattr(options, 'diversity_flops_diff_w')

    @staticmethod
    def add_parser_arguments(parser):
        parser.add_argument('--diversity_top_k', default=4, type=int)
        parser.add_argument('--diversity_max_frac', default=0.1, type=float)
        parser.add_argument('--diversity_prediction_diff_w', default=0., type=float)
        parser.add_argument('--diversity_search_depth_diff_w', default=2e-3, type=float)
        parser.add_argument('--diversity_search_tree_dist_w', default=2e-3, type=float)
        parser.add_argument('--diversity_flops_diff_w', default=1e-10, type=float)
        parser.add_argument('--diversity_combine_type', default='min', choices=['min', 'max', 'avg'])
        return parser


def diverse_top_k(l_mi, l_priority, mi_info, div_opts):
    """
    """
    size = len(l_mi)
    max_idx = min(int(div_opts.max_frac * size), size - 1)
    if max_idx + 1 <= div_opts.top_k:
        return list(range(max_idx + 1))
    
    indices = heapq.nsmallest(max_idx + 1, range(size), key=lambda i : l_priority[i])
    l_top_idx = [indices[0]]
    for _ in range(1, div_opts.top_k):
        min_idx = -1
        min_dp = None
        for idx in indices:
            mi1 = l_mi[idx]
            priority = l_priority[idx]
            l_diff = []
            for selected_idx in l_top_idx:
                mi2 = l_mi[selected_idx]
                diff = prediction_diff(mi1, mi2, mi_info) * div_opts.prediction_diff_w \
                    + search_depth_diff(mi1, mi2, mi_info) * div_opts.search_depth_diff_w \
                    + search_tree_dist(mi1, mi2, mi_info) * div_opts.search_tree_dist_w \
                    + flops_diff(mi1, mi2, mi_info) * div_opts.flops_diff_w
                l_diff.append(diff)
            diversity = compute_diversity(l_diff, div_opts)
            # Smaller value has higher priority. (low val_err)
            # The higher the diveristy, the lower is div_priority, so higher priority
            div_priority = priority - diversity
            if min_dp is None or div_priority < min_dp:
                min_dp = div_priority
                min_idx = idx
        l_top_idx.append(min_idx)
    return l_top_idx


if __name__ == '__main__':
    # for debug
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mi1', type=int)
    parser.add_argument('--mi2', type=int)
    parser.add_argument('--mi_info_npz', type=str)
    args = parser.parse_args()

    mi_info = list(np.load(args.mi_info_npz, encoding='bytes')['mi_info'])
    mi1 = args.mi1
    mi2 = args.mi2
    info1 = mi_info[mi1]
    info2 = mi_info[mi2]

    print("Info 1: {}".format(mi_info[mi1].to_str()))
    print("Info 2: {}".format(mi_info[mi2].to_str()))

    ret = [ prediction_diff(mi1, mi2, mi_info) , search_depth_diff(mi1, mi2, mi_info),
        search_tree_dist(mi1, mi2, mi_info), flops_diff(mi1, mi2, mi_info) ]
    print(ret)