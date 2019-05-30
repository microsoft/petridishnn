import tensorflow as tf
from petridish.info import LayerInfoList

def _get_grad_slice(h_shape):
    if LayerInfoList.HALLU_IDX_IN_MERGE_HALLU == 1:
        return slice(-h_shape, None)
    else:
        return slice(0, h_shape)

def _init_hallu_record(compute_hallu_stats):
    if not compute_hallu_stats:
        return None
    hallu_indices, hallu_outputs, hallu_merge_pts = [], [], []
    return hallu_indices, hallu_outputs, hallu_merge_pts

def _update_hallu_record(
        compute_hallu_stats, hallu_record,
        layer_idx, layer_info_list, layer_dict):
    """
    """
    if not compute_hallu_stats:
        return hallu_record
    hallu_indices, hallu_outputs, hallu_merge_pts = hallu_record
    info = layer_info_list[layer_idx]
    info_id = info.id
    hallu_idx = info.is_candidate
    # Whether this is the last layer of a candidate change.
    is_last = (
        hallu_idx != 0 and
        (layer_idx == len(layer_info_list) - 1 or
         layer_info_list[layer_idx + 1].is_candidate != hallu_idx)
    )
    if is_last:
        # The end goal:
        # 1. hallu_indices contains int of hallu to identify them;
        # 2. hallu_outputs contains tensor of final candidate;
        # 3. hallu_merge_pts contains tensor where candidate merge
        #    into the parent model
        #
        # However, since the merge points are in the future, we
        # record their id for now, and will populate them when we
        # actually construct the merge_points.
        #
        # Furthermore, a hallu may not be directly used by the
        # merge point, but instead first transformed/down-sampled.
        # We also record the hallu id in hallu_outputs at start, and only
        # populate it when we know exactly how it is used.
        hallu_indices.append(hallu_idx)
        hallu_outputs.append(info_id)
        for future_info in layer_info_list[layer_idx:]:
            if info_id in future_info.inputs:
                hallu_merge_pts.append(future_info.id)
                break

    if info_id in hallu_merge_pts:
        for pi, pt in enumerate(hallu_merge_pts):
            if pt == info_id:
                merge_pt = layer_dict[pt]
                # populate the merge point
                hallu_merge_pts[pi] = merge_pt
                # We populate hallu_outputs now that we know how
                # the hallu is used.
                hallu_id = hallu_outputs[pi]
                for in_idx, in_id in enumerate(info.inputs):
                    if in_id == hallu_id:
                        scaled_hallu = merge_pt.op.inputs[in_idx]
                        assert hasattr(scaled_hallu, 'pre_gate_layer'), scaled_hallu
                        hallu_outputs[pi] = scaled_hallu.pre_gate_layer
                        break
    return hallu_indices, hallu_outputs, hallu_merge_pts

def _hallu_stats_graph(compute_hallu_stats, hallu_record, cost, scope):
    if not compute_hallu_stats:
        return
    hallu_indices, hallu_outputs, hallu_merge_pts = hallu_record
    if len(hallu_merge_pts) == 0:
        return
    grads = tf.gradients(cost, hallu_merge_pts)
    stats = []
    scope_prefix = scope.strip("/") + '/'
    for hi, g, h in zip(hallu_indices, grads, hallu_outputs):
        with tf.variable_scope(scope_prefix + 'hallu_stats_{:02d}'.format(hi)):
            h_shape = h.get_shape().as_list()
            g_shape = g.get_shape().as_list()
            if not all([hsh == gsh for hsh, gsh in zip(h_shape, g_shape)]):
                slices = [
                    slice(None, None) if hsh == gsh else _get_grad_slice(hsh)  \
                    for hsh, gsh in zip(h_shape, g_shape)
                ]
                g = g[slices]
            g = tf.reshape(g, shape=[-1])
            h = tf.reshape(h, shape=[-1])
            dotp = tf.reduce_sum(tf.multiply(g, h), name='dot_prod')
            g_l2 = tf.reduce_sum(tf.square(g), name='grad_l2')
            h_l2 = tf.reduce_sum(tf.square(h), name='hallu_l2')
            stats.extend([dotp, g_l2, h_l2])
    return stats

def _hallu_stats_graph_merged(
        compute_hallu_stats, hallu_record, cost, scope,
        n_calls=1, layer_info_list=None):
    if not compute_hallu_stats or hallu_record is None:
        return None
    if n_calls == 1:
        stats = _hallu_stats_graph(compute_hallu_stats, hallu_record, cost, scope)
        return stats
    scope_tmp = scope.strip("/") + '_individual'
    stats = _hallu_stats_graph(compute_hallu_stats, hallu_record, cost, scope_tmp)
    if stats is None:
        return None

    n_vals = len(stats)
    n_val_per_call = n_vals // n_calls
    assert n_val_per_call * n_calls == n_vals, \
        '{} * {} != {}'.format(n_val_per_call, n_calls, n_vals)
    names = get_hallu_stats_output_names(layer_info_list, scope=scope)
    names = list(map(lambda n: n[:-2], names))
    assert len(names) == n_val_per_call, \
        "{} != {}".format(len(names), n_val_per_call)

    # merge stats
    merged_stats = [
        tf.add_n(
            [stats[idx] for idx in range(vi, n_vals, n_val_per_call)],
            name=names[vi]) \
        for vi in range(n_val_per_call)
    ]
    return merged_stats

NUM_STATS_PER_HALLU = 3

def get_hallu_stats_output_names(layer_info_list, scope=None):
    stat_names = ['/dot_prod:0', '/grad_l2:0', '/hallu_l2:0']
    assert len(stat_names) == NUM_STATS_PER_HALLU, stat_names
    names = []
    hallus = set()
    scope_prefix = "" if scope is None else scope + '/'
    for info in layer_info_list:
        hi = info.is_candidate
        if hi > 0 and not hi in hallus:
            hallus.add(hi)
            scope = scope_prefix + 'hallu_stats_{:02d}'.format(hi)
            names.extend([scope + n for n in stat_names])
    return names

def get_net_info_hallu_stats_output_names(net_info):
    hallu_stats_names = []
    for cname in net_info.operable_cell_names:
        layer_info_list = net_info[cname]
        hallu_stats_names.extend(
            get_hallu_stats_output_names(layer_info_list, scope=cname))
    return hallu_stats_names