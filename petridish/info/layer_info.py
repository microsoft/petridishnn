import numpy as np
import json
import copy
import functools

from tensorpack.utils import logger
"""
Layer definitions
"""

class LayerTypes(object):
    # Value regarding stop_gradients not exactly layer types
    STOP_GRADIENT_NONE = 0
    STOP_GRADIENT_HARD = 1
    STOP_GRADIENT_SOFT = 2

    # NOT_EXIST and NOTHING is for layer info sequence.
    NOT_EXIST = 0
    IDENTITY = 1

    # vision
    RESIDUAL_LAYER = 2
    RESIDUAL_BOTTLENECK_LAYER = 3
    CONV_1 = 4
    CONV_3 = 5
    SEPARABLE_CONV_3 = 6
    SEPARABLE_CONV_5 = 7
    SEPARABLE_CONV_3_2 = 23
    SEPARABLE_CONV_5_2 = 24
    SEPARABLE_CONV_7_2 = 25
    DILATED_CONV_3 = 30
    DILATED_CONV_5 = 31

    MAXPOOL_3x3 = 8
    AVGPOOL_3x3 = 9

    # merge
    MERGE_WITH_CAT = 10
    MERGE_WITH_SUM = 11
    MERGE_WITH_AVG = 12
    MERGE_WITH_NOTHING = 13
    MERGE_WITH_MUL = 21
    MERGE_WITH_CAT_PROJ = 22
    MERGE_WITH_SOFTMAX = 27
    MERGE_WITH_WEIGHTED_SUM = 28

    # Gates on hallucinations and the original path
    GATED_LAYER = 14
    ANTI_GATED_LAYER = 17
    NO_FORWARD_LAYER = 26

    # MLP and general operations
    FullyConnected = 15
    MLP_RESIDUAL_LAYER = 16

    # RNN layers
    FC_TANH_MUL_GATE = 18
    FC_RELU_MUL_GATE = 19
    FC_SGMD_MUL_GATE = 20
    FC_IDEN_MUL_GATE = 29

    @staticmethod
    def num_layer_types():
        return 32 # Please update me whenever a new layer is made

    @staticmethod
    def no_param_ops():
        LT = LayerTypes
        return [LT.NOT_EXIST, LT.IDENTITY,
            LT.GATED_LAYER, LT.ANTI_GATED_LAYER, LT.NO_FORWARD_LAYER]

    @staticmethod
    def do_drop_path(op):
        LT = LayerTypes
        return (not LT.has_multi_inputs(op) and not op in LT.no_param_ops())

    @staticmethod
    def sample_layer_type(valid_ops, prob=None):
        n = len(valid_ops)
        if prob is None:
            prob = [1. / n] * n
        else:
            assert len(prob) == n
        return valid_ops[int(np.nonzero(np.random.multinomial(1, prob))[0][0])]

    @staticmethod
    def has_multi_inputs(op):
        return op in [LayerTypes.MERGE_WITH_CAT,
            LayerTypes.MERGE_WITH_SUM, LayerTypes.MERGE_WITH_AVG,
            LayerTypes.MERGE_WITH_MUL, LayerTypes.MERGE_WITH_CAT_PROJ,
            LayerTypes.MERGE_WITH_SOFTMAX, LayerTypes.MERGE_WITH_WEIGHTED_SUM]


class LayerInfo(dict):

    def __init__(self, layer_id, inputs=[], operations=[],
            down_sampling=0, stop_gradient=0, aux_weight=0,
            is_candidate=0, extra_dict=None):
        """
        Args:
            layer_id: a unqiue non-negative int that is the id of the layer.
                When overflows happens, an assertion failure will be triggered

        Kwargs:
            inputs: a list of layer_ids that are the inpu of the layer
            operations: a list of LayerType.<operations> on each of the input
            down_sampling: int/bool. whether this layer will dod down sampling.
            stop_gradient: int or list of int. Each int is 0 or 1, representing whether
                a stop_gradient call is done first to each of the input. If the value is
                just an int, then the same value is applied to all inputs.
            aux_weight: float auxiliary weight on the layer. This is used to mark where the
                prediction happens. E.g., the final layer always should have positive weight.
            is_candidate: int. Candidate id. This represents which candidate this layer is from.
                candidate==0 means the real network, >0 means it is from some hallucination.
        """
        def _assert_is_list(v):
            assert isinstance(v, list), v
        def _assert_is_int_or_bool(v):
            assert isinstance(v, int) or isinstance(v, bool)
        _assert_is_list(inputs)
        _assert_is_list(operations)
        _assert_is_int_or_bool(down_sampling)
        assert isinstance(aux_weight, float) or isinstance(aux_weight, int)
        if len(operations) != len(inputs) + 1:
            assert len(inputs) == len(operations)
        if layer_id is not None:
            assert layer_id >=0, 'Hack failed, we have negative layer idx meow'

        self['id'] = layer_id
        self['inputs'] = inputs
        self['operations'] = operations
        self['down_sampling'] = down_sampling
        self['stop_gradient'] = stop_gradient
        self['aux_weight'] = aux_weight
        self['is_candidate'] = is_candidate
        if isinstance(extra_dict, dict) and len(extra_dict) > 0:
            self['extra_dict'] = extra_dict

    @property
    def id(self):
        return self['id']

    @id.setter
    def id(self, val):
        self['id'] = val

    @property
    def inputs(self):
        return self['inputs']

    @inputs.setter
    def inputs(self, val):
        self['inputs'] = val

    @property
    def operations(self):
        return self['operations']

    @operations.setter
    def operations(self, val):
        self['operations'] = val

    @property
    def down_sampling(self):
        return self['down_sampling']

    @down_sampling.setter
    def down_sampling(self, val):
        self['down_sampling'] = val

    @property
    def stop_gradient(self):
        return self['stop_gradient']

    @stop_gradient.setter
    def stop_gradient(self, val):
        self['stop_gradient'] = val

    @property
    def aux_weight(self):
        return self['aux_weight']

    @aux_weight.setter
    def aux_weight(self, val):
        self['aux_weight'] = val

    @property
    def is_candidate(self):
        return self['is_candidate']

    @is_candidate.setter
    def is_candidate(self, val):
        self['is_candidate'] = val

    @property
    def merge_op(self):
        if len(self['inputs']) == len(self['operations']):
            return LayerTypes.MERGE_WITH_NOTHING
        return self['operations'][-1]

    @merge_op.setter
    def merge_op(self, val):
        if len(self['inputs']) == len(self['operations']):
            self['operations'].appned(val)
        self['operations'][-1] = val

    @property
    def input_ops(self):
        return self['operations'][:len(self['inputs'])]

    @input_ops.setter
    def input_ops(self, val):
        curr = self.get('operations', [])
        if len(curr) <= 1:
            # there is no merge_op before
            self['operations'] = val
        else:
            self['operations'][:-1] = val

    @property
    def extra_dict(self):
        return self.get('extra_dict', None)

    @extra_dict.setter
    def extra_dict(self, val):
        self['extra_dict'] = val

    @staticmethod
    def from_str(ss):
        d = json.loads(ss)
        return LayerInfo.from_json_loads(d)

    def to_str(self):
        return json.dumps(self)

    @staticmethod
    def from_json_loads(d):
        d['layer_id'] = d.pop('id', None)
        return LayerInfo(**d)

    @staticmethod
    def is_input(info):
        return len(info.inputs) == 0


class LayerInfoList(list):
    DELIM = '=^_^='
    LOG_DELIM = '\n'
    n_extra_dim = 4

    ORIG_IDX_IN_MERGE_HALLU = 0
    HALLU_IDX_IN_MERGE_HALLU = 1

    def __init__(self, *args, **kwargs):
        super(LayerInfoList, self).__init__(*args, **kwargs)

    def to_str(self):
        return json.dumps(self)

    def num_inputs(self):
        n_inputs = 0
        for info in self:
            if LayerInfo.is_input(info):
                n_inputs += 1
            else:
                break
        return n_inputs

    @property
    def master(self):
        return self

    def is_end_merge_sum(self):
        LT = LayerTypes
        return self[-1].merge_op in [LT.MERGE_WITH_SUM, LT.MERGE_WITH_AVG]

    def is_end_merge_cat(self):
        LT = LayerTypes
        return self[-1].merge_op in [LT.MERGE_WITH_CAT, LT.MERGE_WITH_CAT_PROJ]

    def sample_cat_hallucinations(self, layer_ops, merge_ops,
        prob_at_layer=None, min_num_hallus=1, hallu_input_choice=None):
        """
        prob_at_layer : probility of having input from a layer. None is translated
            to default, which sample a layer proportional to its ch_dim. The ch_dim
            is computed using self, as we assume the last op is cat, and the cat
            determines the ch_dim.

        """
        assert self[-1].merge_op == LayerTypes.MERGE_WITH_CAT
        n_inputs = self.num_inputs()
        n_final_merge = len(self[-1].inputs)

        if prob_at_layer is None:
            prob_at_layer = np.ones(len(self) - 1)
            prob_at_layer[:n_inputs-1] = n_final_merge
            prob_at_layer[n_inputs-1] = n_final_merge * 1.5
            prob_at_layer = prob_at_layer / np.sum(prob_at_layer)
        assert len(prob_at_layer) >= len(self) - 1
        if len(prob_at_layer) > len(self) - 1:
            logger.warn("sample cell hallu cuts the prob_at_layer to len(info_list) - 1")
            prob_at_layer = prob_at_layer[:len(self)-1]

        # choose inputs
        n_hallu_inputs = 2
        l_hallu = []
        for _ in range(min_num_hallus):
            # replace == True : can connect multiple times to the same layer
            in_idxs = np.random.choice(list(range(len(prob_at_layer))),
                size=n_hallu_inputs, replace=False, p=prob_at_layer)
            in_ids = list(map(lambda idx : self[idx].id, in_idxs))
            main_ops = list(map(int, np.random.choice(layer_ops, size=n_hallu_inputs)))
            merge_op = int(np.random.choice(merge_ops))
            hallu = LayerInfo(layer_id=self[-1].id, inputs=in_ids,
                operations=main_ops + [merge_op])
            l_hallu.append(hallu)
        return l_hallu

    def add_cat_hallucinations(self, l_h,
            final_merge_op=LayerTypes.MERGE_WITH_SUM,
            stop_gradient_val=1,
            hallu_gate_layer=LayerTypes.NO_FORWARD_LAYER):
        assert final_merge_op == LayerTypes.MERGE_WITH_SUM
        assert self[-1].merge_op == LayerTypes.MERGE_WITH_CAT
        l_info = self
        next_id = 0
        max_candidate = 0
        for info in l_info:
            next_id = max(next_id, info.id)
            max_candidate = max(max_candidate, info.is_candidate)
        max_candidate = 0
        l_insert = []
        final_merge_inputs = []
        for h in l_h:
            next_id += 1
            max_candidate += 1
            info = copy.deepcopy(h)
            info.id = next_id
            info.is_candidate = max_candidate
            info.stop_gradient = stop_gradient_val
            l_insert.append(info)
            final_merge_inputs.append(info.id)
        next_id += 1
        ops = [hallu_gate_layer] * len(final_merge_inputs) + [LayerTypes.MERGE_WITH_SUM]
        info = LayerInfo(layer_id=next_id, inputs=final_merge_inputs,
            operations=ops)
        l_insert.append(info)
        l_info[-1].inputs.append(info.id)
        l_info[-1].operations[-1:-1] = [LayerTypes.IDENTITY]
        l_info[-1:-1] = l_insert
        merge_pt = l_info[-1]
        assert not isinstance(merge_pt.stop_gradient, list), \
            "Merge point has invalid stop_gradient: {}".format(merge_pt)
        assert not isinstance(merge_pt.down_sampling, list), \
            "Merge point has invalid down_sampling: {}".format(merge_pt)
        return l_info

    def select_cat_hallucination(self,
            selected, contained=None, hallu_indices=None,
            l_fs_ops=None, l_fs_omega=None):
        assert self[-1].merge_op == LayerTypes.MERGE_WITH_CAT, self[-1].merge_op
        l_info = self
        if isinstance(selected, int):
            selected = [selected]
        if contained is None:
            contained = l_info.contained_hallucination()
        n_contained = len(contained)
        if n_contained == 0:
            return l_info
        if hallu_indices is None:
            hallu_indices = LayerInfoList.sorted_hallu_indices(contained)
        to_keep = []
        for h_idx in reversed(hallu_indices):
            start, end = contained[h_idx]
            if not h_idx in selected:
                l_info[start:end] = []
            else:
                to_keep.append(l_info[end-1].id)
                for x in l_info[start:end]:
                    x.is_candidate = 0
                    x.stop_gradient = 0

        n_to_keep = len(to_keep)
        old_id = l_info[-2].id
        assert old_id == l_info[-1].inputs[-1], '{} != {}'.format(old_id, l_info[-2].id)

        def del_merge_hallu():
            del l_info[-2]

        def del_at_final_cat():
            del l_info[-1].inputs[-1]
            del l_info[-1].operations[-2]
            if isinstance(l_info[-1].down_sampling, list):
                assert len(l_info[-1].down_sampling) == len(l_info[-1].inputs) + 1
                del l_info[-1].down_sampling[-1]
            if isinstance(l_info[-1].stop_gradient, list):
                assert len(l_info[-1].stop_gradient) == len(l_info[-1].inputs) + 1
                del l_info[-1].stop_gradient[-1]

        if n_to_keep > 1:
            for keep_id in to_keep:
                assert keep_id in l_info[-2].inputs, \
                    '{} not in {}'.format(keep_id, l_info[-2].inputs)
            l_info[-2].inputs = to_keep
            l_info[-2].operations = ([LayerTypes.GATED_LAYER] * len(to_keep) +
                [LayerTypes.MERGE_WITH_SUM])

        elif n_to_keep == 0:
            del_merge_hallu()
            del_at_final_cat()

        elif n_to_keep == 1:
            new_id = to_keep[0]
            assert new_id in l_info[-2].inputs, \
                '{} not in {}'.format(new_id, l_info[-2].inputs)
            del_merge_hallu()
            l_info[-1].inputs, l_info[-1].stop_gradient = \
                LayerInfoList._rewire_inputs_post_add(
                    { old_id : new_id }, l_info[-1].inputs, l_info[-1].stop_gradient)
            l_info[-1].operations[-2] = LayerTypes.GATED_LAYER
        return l_info

    def _id_to_idx(self):
        id_to_idx = dict()
        for idx, info in enumerate(self):
            id_to_idx[info.id] = idx
        return id_to_idx

    def _distance_to_idx(self, t_idx, id_to_idx, max_dist=None):
        max_dist = max_dist if max_dist is not None else t_idx + 1
        l_dists = [max_dist + 1] * (t_idx + 1)
        is_fixed = [False] * (t_idx + 1)
        l_dists[-1] = 0
        is_fixed[-1] = True
        queue = [t_idx]
        qidx = 0
        while qidx < len(queue):
            idx = queue[qidx]
            dist = l_dists[idx]
            for in_id in self[idx].inputs:
                in_idx = id_to_idx[in_id]
                if not is_fixed[in_idx]:
                    is_fixed[in_idx] = True
                    l_dists[in_idx] = dist + 1
                    if l_dists[in_idx] < max_dist:
                        queue.append(in_idx)
            qidx += 1
        return l_dists

    def _is_end_of_cell(self, idx):
        """
        Given an index (idx) in the current layer info list,
        determine whether it is an end of a cell in the
        starting of a macro search.

        This method should only be used by a master l_info,
        i.e., a macro search master, or a cell search root/master.

        This method assumes that the original layers have id smaller than
        the last id, which is the last layer of the original
        network as well. Finally, the original layer ids also
        convey the depth of layers.

        (Known bug: we only use the connection pattern and id
        to check this. Ideally one should know this from
        detecting repeatable patterns of directly using
        the original construction function. This is not a
        concern for now as the seed cells are simple).
        """
        l_info = self
        info = l_info[idx]
        if info.id > l_info[-1].id:
            # the info is inserted after seed model
            return False

        # Check input connection patterns.

        def seeded_inputs(_info):
            info_id = _info.id
            return [
                (info_id - in_id, in_op) \
                    for in_id, in_op in zip(_info.inputs, _info.operations) \
                    if in_id < info_id
                ]
        tar_id_ops = seeded_inputs(l_info[-1])
        id_ops = seeded_inputs(info)
        return id_ops == tar_id_ops

    def _sample_output_locations(
            self, min_num_hallus, cell_based):
        """
        Returns:
        A list of indices in the layer info list that will be x_out
        """
        l_info = self
        n_layers = len(l_info)
        n_inputs = l_info.num_inputs()
        LT = LayerTypes

        if cell_based:
            l_x_out = [n_layers - 1] * min_num_hallus
        else:
            # compute the eligible layers.
            indices = [
                idx for idx in range(n_inputs, n_layers) \
                    if l_info._is_end_of_cell(idx)
            ]
            #logger.info("The eligible ones are {}".format(indices))
            n_eligible = len(indices)
            repeats = min_num_hallus // n_eligible
            remainder = np.random.choice(
                indices, min_num_hallus % n_eligible, replace=False
            )
            l_x_out = indices * repeats + list(remainder)
            l_x_out.sort()
        return l_x_out

    def _sample_all_input_ids(self, l_x_out, cell_based):
        """
        1. For cell search, inputs are either in the same
        cell or have lower id. So all layers are possible inputs.
        2. For macro search, we mimic the behavior, and limit
        the inputs to be in the same cell or the output of
        the two previous cells.

        Args:
        l_x_out: a list of indices in the layer info list that are to be x_out
        cell_based (bool) : whether the NAS is for cell or macro search.

        Returns:
        A list of list of layer id that are to be input ids of the x_out in l_x_out.
        """
        l_info = self
        l_inputs = []
        for out_idx in l_x_out:
            if cell_based:
                in_id_pool = [info.id for info in l_info[:out_idx]]
            else:
                # This is the same as cell based search case above,
                # if the cell has two inputs (prev and prev-prev cell output).
                n_origs = 2
                in_same_cell = True
                in_id_pool = []
                for idx in reversed(range(out_idx)):
                    if l_info._is_end_of_cell(idx):
                        in_same_cell = False
                        n_origs -= 1
                        in_id_pool.append(l_info[idx].id)
                        if n_origs == 0:
                            break
                    if in_same_cell:
                        in_id_pool.append(l_info[idx].id)
            l_inputs.append(in_id_pool)
        return l_inputs

    # def _sample_input_ids(self, l_x_out, cell_based):
    #     """
    #     Args:
    #     l_x_out: a list of indices in the layer info list that are to be x_out
    #     cell_based (bool) : whether the NAS is for cell or macro search.

    #     Returns:
    #     A list of list of layer id that are to be input ids of the x_out in l_x_out.
    #     """
    #     l_info = self
    #     n_inputs = l_info.num_inputs()
    #     LT = LayerTypes

    #     # compute the current intensive operations.
    #     n_intensive_ops = 0
    #     for info in l_info[n_inputs:]:
    #         for op in info.operations:
    #             n_intensive_ops += int(
    #                 op in [
    #                     LT.SEPARABLE_CONV_3_2, LT.SEPARABLE_CONV_3_2,
    #                     LT.SEPARABLE_CONV_7_2, LT.CONV_1, LT.CONV_3,
    #                 ]
    #             )
    #     # n_ins_per_out is proportional to number of existing intensive
    #     # operations, and is at least 2 (unless not possible)
    #     n_ins_per_out = max(
    #         2, int(0.5 + float(n_intensive_ops) / len(l_x_out)))
    #     l_inputs = []

    #     # 1. For cell search, inputs are either in the same
    #     # cell or have lower id. So all layers are possible inputs.
    #     # 2. For macro search, we mimic the behavior, and limit
    #     # the inputs to be in the same cell or the output of
    #     # the two previous cells.
    #     max_orig_id = l_info[-1].id
    #     for out_idx in l_x_out:
    #         # Find the most recent two orig idx.
    #         # These are the output of cells.
    #         n_origs = 2
    #         in_same_cell = True
    #         in_id_pool = []
    #         for in_idx in reversed(range(out_idx)):
    #             infoid = l_info[in_idx].id
    #             if infoid <= max_orig_id:
    #                 in_same_cell = False
    #                 n_origs -= 1
    #                 in_id_pool.append(infoid)
    #                 if n_origs == 0:
    #                     break
    #             if in_same_cell:
    #                 in_id_pool.append(infoid)
    #         n_x_ins = min(n_ins_per_out, len(in_id_pool))
    #         inputs = [
    #             int(_x) for _x in np.random.choice(
    #                 in_id_pool, n_x_ins, replace=False)
    #         ]
    #         l_inputs.append(inputs)
    #     return l_inputs


    def sample_sum_hallucinations(
            self, layer_ops, merge_ops, prob_at_layer=None,
            min_num_hallus=1, hallu_input_choice=None,
            cell_based=False):
        """
        Sample hallus
        """
        l_info = self
        LT = LayerTypes
        do_feat_sel = (
            len(merge_ops) == 1 and
            merge_ops[0] == LT.MERGE_WITH_WEIGHTED_SUM)
        l_x_out = self._sample_output_locations(
            min_num_hallus, cell_based)
        l_inputs = self._sample_all_input_ids(l_x_out, cell_based)
        hallus = []
        for out_idx, inputs in zip(l_x_out, l_inputs):
            out_id = l_info[out_idx].id
            if len(inputs) == 0:
                continue
            if not do_feat_sel:
                # sample operations
                main_ops = list(map(int, np.random.choice(layer_ops, len(inputs))))
                merge_op = int(np.random.choice(merge_ops))
            else:
                # feature selection will form all ops
                # op1,..., opk, op1, ..., opk ,....
                main_ops = list(layer_ops) * len(inputs)
                merge_op = LT.MERGE_WITH_WEIGHTED_SUM
                # in1,..., in1, in2, ..., in2, ...
                inputs = [x for l_x in zip(*[inputs] * len(layer_ops)) for x in l_x]
            hallu = LayerInfo(layer_id=out_id, inputs=inputs,
                operations=main_ops + [ merge_op ])
            hallus.append(hallu)
        return hallus

    def select_sum_hallucination(self,
            selected, contained=None, hallu_indices=None,
            l_fs_ops=None, l_fs_omega=None):
        l_info = self
        if isinstance(selected, int):
            selected = [selected]
        if contained is None:
            contained = l_info.contained_hallucination()
        n_contained = len(contained)
        if n_contained == 0:
            return l_info
        if hallu_indices is None:
            hallu_indices = LayerInfoList.sorted_hallu_indices(contained)

        for h_idx_idx in reversed(range(len(hallu_indices))):
            h_idx = hallu_indices[h_idx_idx]
            start, end = contained[h_idx]
            h_layer_id = l_info[end-1].id
            found = False
            # find the exact one user of this candidate.
            for idx in range(end, len(l_info)):
                if h_layer_id in l_info[idx].inputs:
                    found = True
                    break
            assert found, "Did not find id {}".format(h_layer_id)
            to_remove = (
                (not h_idx in selected) or
                (l_info[end-1].merge_op == LayerTypes.MERGE_WITH_WEIGHTED_SUM and
                 len(l_fs_ops[h_idx_idx]) == 0)
            )
            if to_remove:
                l_info[idx] = LayerInfoList._remove_connection_from_id(
                    l_info[idx], h_layer_id)
                l_info[start:end] = []
            else:
                for x in l_info[start:end]:
                    x.is_candidate = 0
                    x.stop_gradient = 0
                # feature selection case:
                hallu_info = l_info[end-1]
                if hallu_info.merge_op == LayerTypes.MERGE_WITH_WEIGHTED_SUM:
                    fs_indices = l_fs_ops[h_idx_idx]
                    fs_omega = l_fs_omega[h_idx_idx]
                    assert len(fs_indices) == len(fs_omega), \
                        'Invalid feat select info i={} omega={}'.format(
                            fs_indices, fs_omega)
                    new_inputs = [
                        hallu_info.inputs[in_idx] for in_idx in fs_indices
                    ]
                    new_operations = [
                        hallu_info.operations[in_idx] for in_idx in fs_indices
                    ]
                    l_info[end-1].inputs = new_inputs
                    l_info[end-1].operations = (
                        #new_operations + [LayerTypes.MERGE_WITH_WEIGHTED_SUM]
                        new_operations + [LayerTypes.MERGE_WITH_CAT_PROJ]
                    )
                    ed = hallu_info.extra_dict
                    ed = dict() if ed is None else ed
                    ed['ops_ids'] = list(map(int, fs_indices))
                    ed['fs_omega'] = list(map(float, fs_omega))
                    l_info[end-1].extra_dict = ed
                # fix future reference to hallu
                for in_idx, in_id in enumerate(l_info[idx].inputs):
                    if in_id == h_layer_id:
                        l_info[idx].operations[in_idx] = LayerTypes.GATED_LAYER
        return l_info

    def add_sum_hallucinations(
            self, l_h,
            final_merge_op=LayerTypes.MERGE_WITH_SUM,
            stop_gradient_val=1,
            hallu_gate_layer=LayerTypes.NO_FORWARD_LAYER):
        """
        Add hallus
        """
        l_info = self
        next_id = 0
        max_candidate = 0
        for info in l_info:
            next_id = max(next_id, info.id)
            max_candidate = max(max_candidate, info.is_candidate)

        last_o_l_info_idx = 0
        for hallu in l_h:
            o_id = hallu.id
            n_layers = len(l_info)
            for idx in range(last_o_l_info_idx, n_layers):
                if l_info[idx].id == o_id:
                    last_o_l_info_idx = idx
                    break
            assert l_info[last_o_l_info_idx].id == o_id, \
                "info {} does not find id {}; start at {}".format(
                    l_info, o_id, last_o_l_info_idx)

            next_id += 1
            max_candidate += 1

            info = copy.deepcopy(hallu)
            info.id = next_id
            info.is_candidate = max_candidate
            info.stop_gradient = stop_gradient_val

            l_info[last_o_l_info_idx].inputs.append(info.id)
            l_info[last_o_l_info_idx].operations[-1:-1] = [hallu_gate_layer]
            merge_pt = l_info[last_o_l_info_idx]
            assert not isinstance(merge_pt.stop_gradient, list), \
                "Merge point has invalid stop_gradient: {}".format(merge_pt)
            assert not isinstance(merge_pt.down_sampling, list), \
                "Merge point has invalid down_sampling: {}".format(merge_pt)
            l_info[last_o_l_info_idx:last_o_l_info_idx] = [info]
        return l_info

    def contained_hallucination(self):
        """
        Args:
        l_info (list of LayerInfo) : a model description that contains hallu

        Returns:
        hallu_idx_to_range : a dict mapping from candidate_id to (start, end) for candidate_id > 0.
            l_info[start:end] contains exactly all info such that info.is_candidate == candidate_id.
        """
        l_info = self
        n_layers = len(l_info)
        prev_candidate = 0
        hallu_idx_to_range = {}
        for idx, info in enumerate(l_info):
            candidate = info.is_candidate
            if candidate == 0:
                prev_candidate = 0
                continue

            is_start = prev_candidate != candidate
            is_last = ((idx + 1 == n_layers) or (l_info[idx+1].is_candidate != candidate))
            prev_candidate = candidate

            if is_start:
                start = idx
            if is_last:
                hallu_idx_to_range[candidate] = (start, idx+1)
        return hallu_idx_to_range

    @staticmethod
    def sorted_hallu_indices(hallu_idx_to_range):
        return sorted(hallu_idx_to_range, key=lambda i : hallu_idx_to_range[i][0])

    @staticmethod
    def _create_info_merge(next_id, h_id, o_id, aux_weight, is_candidate,
            final_merge_op=LayerTypes.MERGE_WITH_SUM,
            hallu_gate_layer=LayerTypes.NO_FORWARD_LAYER):
        """
        Form the LayerInfo for the merge operation between hallu of id h_id and the original
        tensor of id o_id (out_id). The new LayerInfo will have info.id == next_id.
        Return a list of layers used for merging
        Note any change to this function need to be mirrored in _finalize_info_merge
        """
        inputs = [None] * 2
        inputs[LayerInfoList.ORIG_IDX_IN_MERGE_HALLU] = o_id
        inputs[LayerInfoList.HALLU_IDX_IN_MERGE_HALLU] = h_id
        operations = [LayerTypes.IDENTITY] * 2 + [final_merge_op]
        operations[LayerInfoList.HALLU_IDX_IN_MERGE_HALLU] = hallu_gate_layer
        info = LayerInfo(next_id, inputs=inputs, operations=operations,
            aux_weight=aux_weight, is_candidate=is_candidate)
        return [info]

    @staticmethod
    def _finalize_info_merge(l_info, start, end):
        """
        Given an l_info, and start and end idx of a hallu, update the params of l_info[start:end],
        so that they will become part of the model permenantly.
        Note that any change to this function need to be mirrored in _create_info_merge
        """
        for x in l_info[start:end]:
            x.is_candidate = 0
            x.stop_gradient = 0
        anti_gate_idx = LayerInfoList.ORIG_IDX_IN_MERGE_HALLU
        l_info[end-1].operations[anti_gate_idx] = LayerTypes.ANTI_GATED_LAYER
        gate_idx = LayerInfoList.HALLU_IDX_IN_MERGE_HALLU
        l_info[end-1].operations[gate_idx] = LayerTypes.GATED_LAYER
        return l_info

    @staticmethod
    def _rewire_inputs_post_add(prev_id_to_new_id, inputs, stop_gradient):
        """
            Given a dictionary mapping from old id to the new id,
            we swap the occurance of old_id in inputs to new id.
            Return: updated inputs, and stop_gradient
        """
        if not isinstance(stop_gradient, list):
            stop_gradient = [stop_gradient] * len(inputs)
        for ini, inid in enumerate(inputs):
            if inid in prev_id_to_new_id:
                inputs[ini] = prev_id_to_new_id[inid]
                stop_gradient[ini] = 0
        sg_ref = stop_gradient[0]
        for sg in stop_gradient:
            if sg != sg_ref:
                break
        if sg == sg_ref:
            stop_gradient = sg_ref
        return inputs, stop_gradient

    @staticmethod
    def _remove_connection_from_id(info, id_to_remove):
        if not id_to_remove in info.inputs:
            return info
        if isinstance(info.stop_gradient, list):
            assert len(info.stop_gradient) == len(info.inputs), \
                "Invalid info {}".format(info)
        if isinstance(info.down_sampling, list):
            assert len(info.down_sampling) == len(info.inputs), \
                "Invalid info {}".format(info)
        assert len(info.operations) == len(info.inputs) + 1, \
            "Invalid info {}".format(info)

        idx = 0
        while idx < len(info.inputs):
            if info.inputs[idx] == id_to_remove:
                del info.inputs[idx]
                del info.operations[idx]
                if isinstance(info.stop_gradient, list):
                    del info.stop_gradient[idx]
                if isinstance(info.down_sampling, list):
                    del info.down_sampling[idx]
            idx += 1
        return info

    @staticmethod
    def from_str(ss, delim=DELIM):
        # deprecation protection
        if ss[0] == '{' and ss[-1] == '}' and delim in ss:
            return LayerInfoList(map(LayerInfo.from_str, ss.strip().split(delim)))
        l_dict = json.loads(ss)
        return LayerInfoList.from_json_loads(l_dict)

    @staticmethod
    def from_json_loads(l_dict):
        return LayerInfoList(map(LayerInfo.from_json_loads, l_dict))

    @staticmethod
    def to_seq(l_info):
        """
        Transform the layer info list into a sequence to be parsed by recurrent structures.
        """
        n_layers = len(l_info)
        idx_to_id = list(map(lambda info : info.id, l_info))
        id_to_idx = dict(map(lambda idx : (idx_to_id[idx], idx), range(n_layers)))
        seq = []
        # format of appended info
        # 0 : merge type
        # 1 : is_candidate
        # 2 : stop_grad (currently always just 0 or 1)
        # 3 : down_sampling
        n_extra_dim = LayerInfoList.n_extra_dim
        for idx, info in enumerate(l_info):
            input_seq = [LayerTypes.NOT_EXIST] * (idx + n_extra_dim)
            n_inputs = len(info.inputs)
            for input_id, op in zip(info.inputs, info.operations[:n_inputs]):
                input_idx = id_to_idx[input_id]
                input_seq[input_idx] = op
            input_seq[idx] = info.merge_op
            input_seq[idx+1:] = [int(info.is_candidate > 0),
                int(info.stop_gradient),
                int(info.down_sampling),
            ]
            seq.extend(input_seq)
        return seq

    @staticmethod
    def seq_to_img_flag(seq, max_depth=128, make_batch=False):
        img = np.ones([max_depth, max_depth], dtype=int) * LayerTypes.NOT_EXIST
        flag = np.zeros([max_depth, LayerInfoList.n_extra_dim - 1], dtype=int)
        line_len = LayerInfoList.n_extra_dim
        start = 0
        seq_len = len(seq)
        li = 0
        while start < seq_len:
            end_img = start + li + 1
            end_line = start + line_len
            img[li, :li+1] = seq[start:end_img]
            flag[li, :] = seq[end_img:end_line]
            start = end_line
            line_len += 1
            li += 1
        if make_batch:
            img = img.reshape([1, max_depth, max_depth])
            flag = flag.reshape([1, max_depth, LayerInfoList.n_extra_dim - 1])
        return img, flag

    @staticmethod
    def seq_to_hstr(seq, not_exist_str='--'):
        n_extra_dim = LayerInfoList.n_extra_dim
        line_len = n_extra_dim
        seq_len = len(seq)
        start = 0
        ss = []
        while start < seq_len:
            end = start + line_len
            ss.append(' '.join(map(
                lambda x : (not_exist_str if x == LayerTypes.NOT_EXIST
                    else '{:02d}'.format(x)),
                seq[start:end])))
            start = end
            line_len += 1
        return '\n'.join(ss)

    @staticmethod
    def str_to_seq(ss):
        return LayerInfoList.to_seq(LayerInfoList.from_str(ss))
