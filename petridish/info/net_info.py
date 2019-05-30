import numpy as np
import json
import copy
import functools

from tensorpack.utils import logger

from petridish.info.layer_info import LayerInfo, LayerInfoList, LayerTypes

class CellNetworkInfo(dict):

    def __init__(self, master=None, normal=None, reduction=None):
        super(CellNetworkInfo, self).__init__(locals())
        self._cell_names = []
        if master is not None:
            self._cell_names.append('master')
        if normal is not None:
            self._cell_names.append('normal')
        if reduction is not None:
            self._cell_names.append('reduction')

    @property
    def master(self):
        return self.get('master', None)

    @master.setter
    def master(self, val):
        self['master'] = val

    @property
    def normal(self):
        return self.get('normal', None)

    @normal.setter
    def normal(self, val):
        self['normal'] = val

    @property
    def reduction(self):
        return self.get('reduction', None)

    @reduction.setter
    def reduction(self, val):
        self['reduction'] = val

    @property
    def cell_names(self):
        return self._cell_names

    @property
    def operable_cell_names(self):
        if self.normal is None:
            return ['master']
        elif self.reduction is None:
            return ['normal']
        return ['normal', 'reduction']

    def is_cell_based(self):
        return 'normal' in self.operable_cell_names

    def to_str(self):
        return json.dumps({key : self[key] for key in self._cell_names})

    def sample_hallucinations(
            self, layer_ops, merge_ops, prob_at_layer=None,
            min_num_hallus=1, hallu_input_choice=None):
        hallus = dict()
        num_hallu_by_name = dict()
        if len(self.operable_cell_names) > 1:
            assert len(self.operable_cell_names) == 2, \
                self.operable_cell_names
            num_hallu_by_name['normal'] = min_num_hallus
            num_hallu_by_name['reduction'] = min_num_hallus
        else:
            num_hallu_by_name[self.operable_cell_names[0]] = min_num_hallus

        for cname in self.operable_cell_names:
            n_hallus = num_hallu_by_name[cname]
            if n_hallus == 0:
                continue
            if cname == 'master' or self[cname].is_end_merge_sum():
                cell_based = (cname != 'master')
                hallus[cname] = self[cname].sample_sum_hallucinations(layer_ops,
                    merge_ops, prob_at_layer, n_hallus, hallu_input_choice, cell_based)
            else:
                hallus[cname] = self[cname].sample_cat_hallucinations(layer_ops,
                    merge_ops, prob_at_layer, n_hallus, hallu_input_choice)
        return hallus

    def add_hallucinations(self, hallus,
            final_merge_op=LayerTypes.MERGE_WITH_SUM,
            stop_gradient_val=1,
            hallu_gate_layer=LayerTypes.NO_FORWARD_LAYER):
        for cname in hallus:
            args = [hallus[cname], final_merge_op, stop_gradient_val, hallu_gate_layer]
            if cname == 'master' or self[cname].is_end_merge_sum():
                self[cname].add_sum_hallucinations(*args)
            else:
                self[cname].add_cat_hallucinations(*args)
        return self

    def contained_hallucination(self):
        hallu_locs = dict()
        for ci, cname in enumerate(self.operable_cell_names):
            # candidate id to (start, end)
            hid_to_range = self[cname].contained_hallucination()
            for hid in hid_to_range:
                hallu_locs[(ci, hid)] = hid_to_range[hid]
        return hallu_locs

    def sorted_hallu_indices(self, hallu_locs):
        # sort by ci, then location in list
        return sorted(hallu_locs, key=lambda ci_hid : (ci_hid[0], hallu_locs[ci_hid][0]))

    def separate_hallu_info_by_cname(self, contained, hallu_indices, l_fs_ops, l_fs_omega):
        """
        Args:
        contained : a dict from (ci, hid) to (start, end) in self[operable_cnames[ci]]
        hallu_indices : list of (ci, hid), in order by sorted_hallu_indices
        l_fs_ops : list of list of int indices that represent the order of importance of
            input op of the hallu feature selection. The first level list is in the
            same order as hallu_indices (sorted by (ci,hid) ). These indices are the
            ones that are chosen by each hallu.
        l_fs_omega : list of list of float value that represent the importance value
            whose abosolute value is in decreasing value. The first level is the in
            the same order as l_op_indices and hallu_indices
            These value are associated with the chosen operations.
        """
        cell_names = self.operable_cell_names
        # first break the info by cname so that we can call cell/layerInfoList level api.
        # dictionary from hid to location (start, end)
        lil_contained = { cname : dict() for cname in cell_names }
        for ci_hid in contained:
            ci, hid = ci_hid
            cname = cell_names[ci]
            lil_contained[cname][hid] = contained[ci_hid]
        # hid in sorted order for each cname
        lil_h_indices = { cname : [] for cname in cell_names }
        for ci_hid in hallu_indices:
            ci, hid = ci_hid
            cname = cell_names[ci]
            lil_h_indices[cname].append(hid)

        # Feature selection info
        if l_fs_ops is None or len(l_fs_ops) == 0:
            lil_fs_ops = { cname : None for cname in cell_names }
            lil_fs_omega = { cname : None for cname in cell_names }
        else:
            lil_fs_ops = { cname : [] for cname in cell_names }
            lil_fs_omega = { cname : [] for cname in cell_names }
            for ci_hid, fs_ops, fs_omega in zip(hallu_indices, l_fs_ops, l_fs_omega):
                ci, hid = ci_hid
                cname = cell_names[ci]
                lil_fs_ops[cname].append(fs_ops)
                lil_fs_omega[cname].append(fs_omega)
        return (lil_contained, lil_h_indices, lil_fs_ops, lil_fs_omega)

    def select_hallucination(self, selected, separated_hallu_info):
        """
        selected : list of (ci, hid)
        """
        cell_names = self.operable_cell_names
        # selected hid for each cname
        lil_selected = { cname : [] for cname in cell_names }
        for ci_hid in selected:
            ci, hid = ci_hid
            cname = cell_names[ci]
            lil_selected[cname].append(hid)

        (lil_contained, lil_h_indices, lil_fs_ops, lil_fs_omega) = separated_hallu_info

        # Invoke LayerInfoList select
        for cname in cell_names:
            lil_args = [lil_selected[cname], lil_contained[cname],
                lil_h_indices[cname], lil_fs_ops[cname], lil_fs_omega[cname]
            ]
            if cname == 'master' or self[cname].is_end_merge_sum():
                self[cname] = self[cname].select_sum_hallucination(*lil_args)
            else:
                self[cname] = self[cname].select_cat_hallucination(*lil_args)
        return self

    @staticmethod
    def calc_reduction_layers(num_cells, num_reduction_layers, num_init_reductions):
        """
        Compute true_cell_idx of reduction layers
        """
        reduction_layers = list(range(num_init_reductions))
        for pool_num in range(1, num_reduction_layers + 1):
            layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
            layer_num = int(layer_num) + pool_num - 1 + num_init_reductions
            reduction_layers.append(layer_num)
        return reduction_layers

    @staticmethod
    def default_master(n_normal_inputs=2, n_reduction_inputs=2,
            num_cells=18, num_reduction_layers=2, num_init_reductions=0,
            skip_reduction_layer_input=0, use_aux_head=1):
        reduction_layers = CellNetworkInfo.calc_reduction_layers(
                num_cells, num_reduction_layers, num_init_reductions)
        master = LayerInfoList()
        layer_id = 0
        n_inputs = n_normal_inputs if num_init_reductions == 0 else n_reduction_inputs
        for _ in range(n_inputs):
            master.append(LayerInfo(layer_id=layer_id))
            layer_id += 1

        # true_num_cells counts cells from the first non-input with 0-based index
        true_num_cells = num_cells + num_init_reductions + num_reduction_layers
        for ci in range(true_num_cells):
            info = LayerInfo(layer_id)
            if ci in reduction_layers:
                info.inputs = list(range(layer_id - n_reduction_inputs, layer_id))
                n_in = len(info.inputs)
                info.operations = [LayerTypes.IDENTITY] * n_in + ['reduction']
                info.down_sampling = 1
            else:
                if (skip_reduction_layer_input and ci-1 in reduction_layers and
                        ci > num_init_reductions):
                    # imagenet : do not take the input of regular reduction as skip connection.
                    info.inputs = (list(range(layer_id - n_normal_inputs - 1, layer_id - 2)) +
                        [layer_id - 1])
                else:
                    info.inputs = list(range(layer_id - n_normal_inputs, layer_id))
                n_in = len(info.inputs)
                info.operations = [LayerTypes.IDENTITY] * n_in + ['normal']
            master.append(info)
            layer_id += 1

        # aux_weight at the last cell before the last reduction
        if use_aux_head and len(reduction_layers) > 0:
            master[reduction_layers[-1] - 1 + n_inputs].aux_weight = 0.4
        master[-1].aux_weight = 1.0
        return master

    @staticmethod
    def from_str(ss):
        json_data = json.loads(ss)
        return CellNetworkInfo.from_json_loads(json_data)

    @staticmethod
    def from_json_loads(json_data):
        net_info = CellNetworkInfo()
        if isinstance(json_data, list):
            net_info['master'] = LayerInfoList.from_json_loads(json_data)
            net_info.cell_names.append('master')
        else:
            for key in ['master', 'normal', 'reduction']:
                jd = json_data.get(key, None)
                if jd:
                    net_info[key] = LayerInfoList.from_json_loads(jd)
                    net_info.cell_names.append(key)
        return net_info

    @staticmethod
    def to_seq(rmi):
        return None

    @staticmethod
    def seq_to_img_flag(seq, max_depth=128, make_batcch=False):
        return None

    @staticmethod
    def seq_to_hstr(rmi, not_exist_str='--'):
        return None

    @staticmethod
    def str_to_seq(ss):
        return CellNetworkInfo.to_seq(CellNetworkInfo.from_str(ss))

def net_info_from_str(ss):
    if ss[0] == '{' and ss[-1] == '}' and LayerInfoList.DELIM in ss:
        # this is for backward compatibility
        ss = '[ ' + ss.replace(LayerInfoList.DELIM, ' , ') + ' ]'
    json_data = json.loads(ss)
    return CellNetworkInfo.from_json_loads(json_data)

"""
Examples for resnet, nasnet-a
"""
def separable_resnet_cell_info(next_id=0, input_ids=[0, 1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    LT = LayerTypes
    l_info = LayerInfoList(
        [
            LayerInfo(input_ids[0]),
            LayerInfo(input_ids[1]),
            LayerInfo(
                next_id+2,
                inputs=[input_ids[1], input_ids[1]],
                operations=[
                    LT.SEPARABLE_CONV_3_2,
                    LT.IDENTITY,
                    LT.MERGE_WITH_SUM
                ]
            )
        ])
    return ensure_end_merge(l_info, end_merge)
separable_resnet_cell_info.n_inputs = 2

def basic_resnet_cell_info(next_id=0, input_ids=[0, 1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    LT = LayerTypes
    l_info = LayerInfoList(
        [
            LayerInfo(input_ids[0]),
            LayerInfo(input_ids[1]),
            LayerInfo(
                next_id+2,
                inputs=[input_ids[1]],
                operations=[
                    LT.CONV_3,
                    LT.MERGE_WITH_NOTHING
                ]
            ),
            LayerInfo(
                next_id+3,
                inputs=[next_id+2, input_ids[1]],
                operations=[
                    LT.CONV_3,
                    LT.IDENTITY,
                    LT.MERGE_WITH_SUM
                ]
            )
        ])
    return ensure_end_merge(l_info, end_merge)
basic_resnet_cell_info.n_inputs = 2

def fully_connected_resnet_cell_info(next_id=0, input_ids=[0, 1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    LT = LayerTypes
    l_info = LayerInfoList(
        [
            LayerInfo(input_ids[0]),
            LayerInfo(input_ids[1]),
            LayerInfo(
                next_id+2, inputs=[input_ids[1]],
                operations=[LT.FC_SGMD_MUL_GATE, LT.MERGE_WITH_SUM]
            )
        ])
    return ensure_end_merge(l_info, end_merge)
fully_connected_resnet_cell_info.n_inputs = 2

def fully_connected_rnn_base_cell_info(next_id=0, input_ids=[0,1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    """
    See implementation of PetridishRNNCell to see an example of
    how this list of info is used.
    The first two info are x_and_h and init_layer, which is a
    projected x_and_h multiplied with gate.
    The rest of layers use specified operation to morph the layers.

    This is DARTS v2.
    """
    LT = LayerTypes
    l_info = LayerInfoList(
        [
            LayerInfo(input_ids[0]), # next_id + 0
            LayerInfo(input_ids[1]), # next_id + 1
            LayerInfo(
                next_id+2, inputs=[input_ids[1]],
                operations=[LT.FC_SGMD_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+3, inputs=[next_id+2],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+4, inputs=[next_id+2],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+5, inputs=[next_id+2],
                operations=[LT.FC_IDEN_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+6, inputs=[next_id+3],
                operations=[LT.FC_TANH_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+7, inputs=[next_id+6],
                operations=[LT.FC_SGMD_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+8, inputs=[next_id+4],
                operations=[LT.FC_TANH_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+9, inputs=[next_id+6],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+10,
                inputs=[
                    next_id+1, next_id+2, next_id+3,
                    next_id+4, next_id+5, next_id+6,
                    next_id+7, next_id+8, next_id+9,],
                operations=[
                    LT.IDENTITY, LT.IDENTITY, LT.IDENTITY,
                    LT.IDENTITY, LT.IDENTITY, LT.IDENTITY,
                    LT.IDENTITY, LT.IDENTITY, LT.IDENTITY,
                    LT.MERGE_WITH_AVG]
            )
        ])
    return l_info
fully_connected_rnn_base_cell_info.n_inputs = 2

def darts_rnn_base_cell_info(
        next_id=0, input_ids=[0,1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    """
    See implementation of PetridishRNNCell to see an example of
    how this list of info is used.
    The first two info are x_and_h and init_layer, which is a
    projected x_and_h multiplied with gate.
    The rest of layers use specified operation to morph the layers.

    This is DARTS from the paper writing
    """
    LT = LayerTypes
    l_info = LayerInfoList(
        [
            LayerInfo(input_ids[0]), # next_id + 0
            LayerInfo(input_ids[1]), # next_id + 1
            LayerInfo(
                next_id+2, inputs=[input_ids[1]],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+3, inputs=[next_id+2],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+4, inputs=[next_id+3],
                operations=[LT.FC_TANH_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+5, inputs=[next_id+4],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+6, inputs=[next_id+5],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+7, inputs=[next_id+2],
                operations=[LT.FC_IDEN_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+8, inputs=[next_id+6],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+9, inputs=[next_id+2],
                operations=[LT.FC_RELU_MUL_GATE, LT.MERGE_WITH_SUM]
            ),
            LayerInfo(
                next_id+10,
                inputs=[
                    next_id+1, next_id+2, next_id+3,
                    next_id+4, next_id+5, next_id+6,
                    next_id+7, next_id+8, next_id+9,],
                operations=[
                    LT.IDENTITY, LT.IDENTITY, LT.IDENTITY,
                    LT.IDENTITY, LT.IDENTITY, LT.IDENTITY,
                    LT.IDENTITY, LT.IDENTITY, LT.IDENTITY,
                    LT.MERGE_WITH_AVG]
            )
        ])
    return l_info
darts_rnn_base_cell_info.n_inputs = 2


def resnet_bottleneck_cell_info(down_sampling=0):
    raise NotImplementedError("not implemented due to changing filter sizes")

def nasneta_cell_info(
        next_id=0, input_ids=[0, 1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    LT = LayerTypes
    l_info = LayerInfoList()
    l_info.extend([
        LayerInfo(input_ids[0]),
        LayerInfo(input_ids[1]), # most recent layer
        LayerInfo(next_id+2, inputs=[input_ids[1], input_ids[0]],
            operations=[LT.SEPARABLE_CONV_5_2, LT.SEPARABLE_CONV_3_2, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+3, inputs=[input_ids[0], input_ids[0]],
            operations=[LT.SEPARABLE_CONV_5_2, LT.SEPARABLE_CONV_3_2, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+4, inputs=[input_ids[1], input_ids[0]],
            operations=[LT.AVGPOOL_3x3, LT.IDENTITY, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+5, inputs=[input_ids[0], input_ids[0]],
            operations=[LT.AVGPOOL_3x3, LT.AVGPOOL_3x3, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+6, inputs=[input_ids[1], input_ids[1]],
            operations=[LT.SEPARABLE_CONV_3_2, LT.IDENTITY, LT.MERGE_WITH_SUM]),
    ])
    l_info.append(cat_unused(l_info, next_id+7, end_merge))
    return l_info
nasneta_cell_info.n_inputs = 2

def nasnata_reduction_cell_info(
        next_id=0, input_ids=[0, 1],
        end_merge=LayerTypes.MERGE_WITH_CAT):
    LT = LayerTypes
    l_info = LayerInfoList()
    l_info.extend([
        LayerInfo(input_ids[0]),
        LayerInfo(input_ids[1]), # most recent layer
        LayerInfo(next_id+2, inputs=[input_ids[1], input_ids[0]],
            operations=[LT.SEPARABLE_CONV_5_2, LT.SEPARABLE_CONV_7_2, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+3, inputs=[input_ids[1], input_ids[0]],
            operations=[LT.MAXPOOL_3x3, LT.SEPARABLE_CONV_7_2, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+4, inputs=[input_ids[1], input_ids[0]],
            operations=[LT.AVGPOOL_3x3, LT.SEPARABLE_CONV_5_2, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+5, inputs=[next_id+3, next_id+2],
            operations=[LT.IDENTITY, LT.AVGPOOL_3x3, LT.MERGE_WITH_SUM]),
        LayerInfo(next_id+6, inputs=[next_id+2, input_ids[1]],
            operations=[LT.SEPARABLE_CONV_3_2, LT.MAXPOOL_3x3, LT.MERGE_WITH_SUM]),
    ])
    l_info.append(cat_unused(l_info, next_id+7, end_merge))
    return l_info
nasnata_reduction_cell_info.n_inputs = 2

def cat_unused(layer_info_list, layer_id, end_merge):
    is_used = set()
    layer_dict = set()
    for li, info in enumerate(layer_info_list):
        if LayerInfo.is_input(info):
            is_used.add(li)
            continue
        for in_id in info.inputs:
            is_used.add(in_id)
        layer_dict.add(li)

    inputs = [info.id for info in layer_info_list if not info.id in is_used]
    ops = [LayerTypes.IDENTITY] * (len(inputs) + 1)
    ops[-1] = end_merge
    info = LayerInfo(layer_id, inputs=inputs, operations=ops)
    return info

def ensure_end_merge(l_info, end_merge):
    LT = LayerTypes
    assert len(l_info) > 0, l_info
    if end_merge != l_info[-1].merge_op:
        last_id = l_info[-1].id
        l_info.append(LayerInfo(last_id + 1, inputs=[last_id],
            operations=[LT.IDENTITY, end_merge]))
    return l_info

def replace_wsum_with_catproj(net_info):
    l_lil = []
    if net_info.is_cell_based():
        if net_info.normal:
            l_lil.append(net_info.normal)
        if net_info.reduction:
            l_lil.append(net_info.reduction)
    else:
        l_lil.append(net_info.master)

    for lil in l_lil:
        for info in lil:
            if info.merge_op == LayerTypes.MERGE_WITH_WEIGHTED_SUM:
                info.merge_op = LayerTypes.MERGE_WITH_CAT_PROJ
    return net_info

def add_aux_weight(net_info, aux_weight=0.4):
    last_orig_id = net_info.master[-1].id
    has_pass_reduction = False
    for info in reversed(net_info.master):
        if info.down_sampling:
            has_pass_reduction = True
        elif has_pass_reduction and info.id < last_orig_id:
            info.aux_weight = aux_weight
            break
    return net_info

def net_info_cifar_to_ilsvrc(net_info, s_type, use_latest_input=False):
    # if there are reduction cell, then use reduction cell twice.
    # if there are no reduction cell, then use s_type=imagenet
    assert isinstance(net_info, CellNetworkInfo), \
        "{} is not CellNetworkInfo.".format(net_info)

    # number of reduction that already happened.
    if s_type == 'imagenet':
        n_stem_reductions = 2
    elif s_type == 'basic':
        n_stem_reductions = 0
    elif s_type == 'conv3' or s_type == 'conv7':
        n_stem_reductions = 1
    n_model_reductions = sum([info.down_sampling for info in net_info.master])
    # number of reduction required at start
    n_extra_reductions = 5 - n_model_reductions - n_stem_reductions
    if n_extra_reductions <= 0:
        return net_info

    next_lid = max([info.id for info in net_info.master])
    n_inputs = 2
    layer_ids = [net_info.master[n_inputs - 1].id] * n_inputs
    is_cell_based = bool(net_info.get('reduction', None))
    l_to_insert = []
    for _ in range(n_extra_reductions):
        next_lid += 1
        if is_cell_based:
            operations = [LayerTypes.IDENTITY] * n_inputs + ['reduction']
        else:
            operations = [
                LayerTypes.SEPARABLE_CONV_7_2,
                LayerTypes.IDENTITY,
                LayerTypes.MERGE_WITH_SUM
            ]
        info = LayerInfo(
            next_lid,
            inputs=layer_ids[-n_inputs:],
            operations=operations,
            down_sampling=1)
        layer_ids.append(next_lid)
        l_to_insert.append(info)

    # rewire later layers that use inputs directly
    def mapped_input(old_idx):
        if use_latest_input:
            return layer_ids[n_extra_reductions + n_inputs - 1]
        return layer_ids[n_extra_reductions + old_idx]

    remap_dict = dict(
        [(info.id, mapped_input(idx)) \
            for idx, info in enumerate(net_info.master[:n_inputs])])
    for info in net_info.master:
        for idx, inid in enumerate(info.inputs):
            newid = remap_dict.get(inid, None)
            if newid is not None:
                info.inputs[idx] = newid
    # insertion
    net_info.master[n_inputs:n_inputs] = l_to_insert
    return net_info


def increase_net_info_size(net_info, multiplier=2):
    """
    Increase the size of a macro net_info to be multiplier
    times of the original size. This is used after macro
    searching on small models to enable deeper models.

    Algorithm:
    1. We first find where the cells start and end, using
    _is_end_of_cell(). Check the assumptions there.
    2. For each cell, the inner cell connections are kept
    the same.
    3. The connections to previous end of cells are considered
    as relative.
    4. Each cell is repeated multiplier number of times,
    the repeats are inserted before the real one,

    Args:
    net_info : a CellNetworkInfo for macro search.
    multiplier (int or list of int) :
        If it is int, the number of times each normal cell is repeated.
        If it is a list, it is the periodic multiplier applied to each normal cell.

    return:
    A modified original net_info. Note that the original is changed.
    """
    l_info = net_info.master
    n_inputs = l_info.num_inputs()
    end_cell_indices = list(range(n_inputs))
    orig_end_cell_ids = [
        l_info[idx].id for idx in end_cell_indices
    ]
    next_id = 0
    for info in l_info:
        next_id = max(next_id, info.id)
    idx = start = n_inputs
    id_to_idx = dict()
    normal_cnt = 0
    if isinstance(multiplier, int):
        multiplier = [multiplier]
    while idx < len(l_info):
        # using while loop as l_info is getting longer
        id_to_idx[l_info[idx].id] = idx
        if not l_info._is_end_of_cell(idx):
            idx += 1
            continue
        n_copies = 0
        if not l_info[idx].down_sampling:
            n_copies = multiplier[normal_cnt % len(multiplier)] - 1
            normal_cnt += 1

        cell_size = idx - start + 1

        # make copies.
        for cp_idx in range(n_copies):
            l_info[start:start] = copy.deepcopy(l_info[start:idx+1])
            for info in l_info[start:idx+1]:
                next_id += 1
                info.id = next_id
                inputs = info.inputs
                for in_idx, in_id in enumerate(inputs):
                    if in_id in id_to_idx.keys():
                        idx_in_l_info =  id_to_idx[in_id] + cp_idx * cell_size
                    else:
                        n_prev = None
                        for _i, _id in enumerate(reversed(orig_end_cell_ids)):
                            if _id == in_id:
                                n_prev = _i + 1
                                break
                        idx_in_l_info = end_cell_indices[-n_prev]
                    inputs[in_idx] = l_info[idx_in_l_info].id
                info.inputs = inputs
                # copied cells never produces aux predictions.
                info.aux_weight = 0
            end_cell_indices.append(idx)
            start = idx + 1
            idx += cell_size

        # modify the original for the cell connections
        for info in l_info[start:idx+1]:
            inputs = info.inputs
            for in_idx, in_id in enumerate(inputs):
                if in_id not in id_to_idx.keys():
                    n_prev = None
                    for _i, _id in enumerate(reversed(orig_end_cell_ids)):
                        if _id == in_id:
                            n_prev = _i + 1
                            break
                    inputs[in_idx] = l_info[end_cell_indices[-n_prev]].id
            info.inputs = inputs

        end_cell_indices.append(idx)
        orig_end_cell_ids.append(l_info[idx].id)
        id_to_idx = dict()
        idx = start = end_cell_indices[-1] + 1
    #end while
    net_info.master =  l_info
    return net_info
