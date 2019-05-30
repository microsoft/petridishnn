"""
Controller holds constants regarding how the exploration/exploitation is done.
It also contains the logics of these actions.
"""

import copy
import numpy as np
import bisect
import os
import shutil
import time
from functools import reduce

import tensorflow as tf
from tensorpack.utils import logger

from petridish.app.options import scale_int_val_with_gpu
from petridish.nas_control.critic import (
    critic_train, critic_predictor, CriticTypes, critic_feature)
from petridish.info import (
    LayerTypes, LayerInfo,
    LayerInfoList, CellNetworkInfo, net_info_from_str,
    separable_resnet_cell_info, nasneta_cell_info, basic_resnet_cell_info,
    nasnata_reduction_cell_info, fully_connected_resnet_cell_info,
    fully_connected_rnn_base_cell_info, darts_rnn_base_cell_info)
from petridish.model import (
    PetridishRNNInputWiseModel, PetridishRNNSingleOutputModel,
    RecognitionModel, MLPModel)
from petridish.nas_control.queue import (
    PetridishQueue, PetridishHeapQueue, PetridishSortedQueue,
    IDX_CNT, IDX_PV, IDX_PQE)
from petridish.nas_control.queue_diversity import DiversityOptions
from petridish.app.directory import (
    _mi_to_dn, _ci_to_dn, _latest_ci, _all_mi, _mi_info_save_fn)
from petridish.utils.geometry import _convex_hull_from_points

__all__ = [
    'ControllerTypes', 'PetridishController',
    'RecognitionController', 'MLPController', 'QueueSortMethods',
    'Q_PARENT', 'Q_HALLU', 'Q_CHILD', 'PetridishRecover', 'RNNController'
]

class ControllerTypes(object):
    RECOGNITION = 0
    MLP = 1
    RNN_SINGLE = 2
    RNN_PER_STEP = 3

    @staticmethod
    def type_idx_to_child_model_cls(t):
        model_cls = None
        if t == ControllerTypes.RECOGNITION:
            model_cls = RecognitionModel
        elif t == ControllerTypes.MLP:
            model_cls = MLPModel
        elif t == ControllerTypes.RNN_PER_STEP:
            model_cls = PetridishRNNInputWiseModel
        elif t == ControllerTypes.RNN_SINGLE:
            model_cls = PetridishRNNSingleOutputModel
        return model_cls

    @staticmethod
    def type_idx_to_controller_cls(t):
        ctrl_cls = None
        if t == ControllerTypes.RECOGNITION:
            ctrl_cls = RecognitionController
        elif t == ControllerTypes.MLP:
            ctrl_cls = MLPController
        elif t == ControllerTypes.RNN_PER_STEP:
            ctrl_cls = RNNController
        elif t == ControllerTypes.RNN_SINGLE:
            ctrl_cls = RNNController
        return ctrl_cls

class QueueSortMethods(object):
    RANDOM = 0
    GREEDY = 1
    PREDICT = 2
    DIVERSE_GREEDY = 3
    FIFO = 4
    FILO = 5
    GREEDY_STATS = 6
    CONVEX_HULL = 7 # a queue parent only method
    KITCHEN_SINK = 8
    KITCHEN_SINK_V2 = 9
    KITCHEN_SINK_V3 = 10
    KITCHEN_SINK_V4 = 11


    @staticmethod
    def to_str(t):
        QM = QueueSortMethods
        if t == QM.RANDOM:
            return 'rand'
        elif t == QM.GREEDY:
            return 'greed'
        elif t == QM.PREDICT:
            return 'pred'
        elif t == QM.DIVERSE_GREEDY:
            return 'cover'
        elif t == QM.FIFO:
            return 'fifo'
        elif t == QM.FILO:
            return 'filo'
        elif t == QM.GREEDY_STATS:
            return 'stats'
        elif t == QM.CONVEX_HULL:
            return 'hull'
        elif t == QM.KITCHEN_SINK:
            return 'sink'
        elif t == QM.KITCHEN_SINK_V2:
            return 'sink_v2'
        elif t == QM.KITCHEN_SINK_V3:
            return 'sink_v3'
        elif t == QM.KITCHEN_SINK_V4:
            return 'sink_v4'
        raise ValueError("Unknown QueueSortMethod")

Q_PARENT = 'q_parent'
Q_HALLU = 'q_hallu'
Q_CHILD = 'q_child'

class PetridishController(object):

    def __init__(self, options):
        super(PetridishController, self).__init__()
        self.max_growth = options.max_growth
        self.valid_operations = [LayerTypes.IDENTITY]

        self.q_parent_method = options.q_parent_method
        self.q_hallu_method = options.q_hallu_method
        self.q_child_method = options.q_child_method
        self.q_parent_convex_hull_eps = options.q_parent_convex_hull_eps
        self.do_convex_hull_increase = options.do_convex_hull_increase
        self.critic_type = options.critic_type
        self.controller_seq_length = options.controller_seq_length
        self.controller_max_depth = options.controller_max_depth
        self.controller_batch_size = options.controller_batch_size
        self.controller_dropout_kp = options.controller_dropout_kp
        self.controller_train_every = options.controller_train_every
        self.controller_save_every = options.controller_save_every
        self.lstm_size = options.lstm_size
        self.num_lstms = options.num_lstms
        self.set_netmorph_method(options.netmorph_method)

        self.hallu_final_merge_op = options.hallu_final_merge_op
        self.n_hallu_procs = scale_int_val_with_gpu(
            options.n_hallu_procs_per_gpu, options.nr_gpu)
        self.n_model_procs = scale_int_val_with_gpu(
            options.n_model_procs_per_gpu, options.nr_gpu)
        self.n_critic_procs = scale_int_val_with_gpu(
            options.n_critic_procs_per_gpu, options.nr_gpu)
        self.critic_train_epoch = options.critic_train_epoch
        self.critic_init_lr = options.critic_init_lr
        # Number of hallucination job a parent will generate, if there are
        # many idle workers.
        self.n_hallu_per_parent_on_idle = 2

        self.queue_names = [Q_PARENT, Q_HALLU, Q_CHILD]
        self.queues = None
        self.n_hallu_stats = { Q_PARENT : 1, Q_HALLU : 0, Q_CHILD : 1 }
        self.queue_cls = {
            Q_PARENT : PetridishHeapQueue,
            Q_HALLU : PetridishHeapQueue,
            Q_CHILD : PetridishHeapQueue
        }

        self.predictors = dict()

        self._is_queue_init = False
        self._is_critic_init = False

        # conditional initialization:
        method = self.q_parent_method
        if method == QueueSortMethods.DIVERSE_GREEDY:
            self.diverse_parent = []
            self.div_opts = DiversityOptions(options)
        elif method == QueueSortMethods.CONVEX_HULL:
            self.queue_cls[Q_PARENT] = PetridishSortedQueue

        self.recover = PetridishRecover(self, options)
        self.options = options

    def set_netmorph_method(self, netmorph_method):
        if netmorph_method == 'hard':
            self.stop_gradient_val = LayerTypes.STOP_GRADIENT_HARD
            self.hallu_gate_layer = LayerTypes.NO_FORWARD_LAYER
        elif netmorph_method == 'soft':
            self.stop_gradient_val = LayerTypes.STOP_GRADIENT_NONE
            self.hallu_gate_layer = LayerTypes.GATED_LAYER
        elif netmorph_method == 'soft_gateback':
            self.stop_gradient_val = LayerTypes.STOP_GRADIENT_SOFT
            self.hallu_gate_layer = LayerTypes.GATED_LAYER
        else:
            raise ValueError("Unknown netmorph method {}".format(netmorph_method))

    def init_queues(self):
        if self._is_queue_init:
            return self.queue_names, self.queues
        self._is_queue_init = True

        queues = [
            self.queue_cls[name](name) for name in self.queue_names
        ]
        self.q_parent, self.q_hallu, self.q_child = queues
        self.queues = dict([(q.name, q) for q in queues])
        return self.queue_names, self.queues


    def init_predictors(self, log_dir_root, model_dir_root):
        if self._is_critic_init:
            return
        self._is_critic_init = True
        qname_to_ci = dict()
        for vs_name in self.queue_names:
            ci = _latest_ci(log_dir_root, model_dir_root, vs_name)
            model_dir = None if ci is None else _ci_to_dn(model_dir_root, ci, vs_name)
            qname_to_ci[vs_name] = -1 if ci is None else ci
            self.predictors[vs_name] = critic_predictor(self, model_dir, vs_name)
        return qname_to_ci


    def update_predictor(self, model_dir, vs_name):
        ckpt = tf.train.latest_checkpoint(model_dir)
        if ckpt:
            logger.info("predictor {} restore from {}".format(vs_name, ckpt))
            self.predictors[vs_name].restore(ckpt)


    def predict_batch(self, l_feats, vs_name):
        ret = []
        for feat in l_feats:
            ret.append(self.predict_one(feat, vs_name))
        #logger.info("predict_batch has result {}".format(ret))
        return ret


    def predict_one(self, pred_feat, vs_name):
        p = self.predictors[vs_name]
        ret = p(*pred_feat)[0][0]
        return ret


    def update_queue(self, queue, mi_info):
        l_feats = [
            critic_feature(self, mi_info, pqe.model_iter, None) \
                for pqe in queue.all_as_generator()
        ]
        l_priority = self.predict_batch(l_feats, vs_name=queue.name)
        queue.update(l_priority=l_priority)
        return queue


    def add_one_to_queue(self, queue, mi_info, model_iter, l_info=None):
        vs_name = queue.name
        info = mi_info[model_iter]
        pv = None  # priority value
        QM = QueueSortMethods
        # Special cases
        if vs_name == Q_PARENT:
            method = self.q_parent_method
            if method in [QM.GREEDY, QM.DIVERSE_GREEDY]:
                pv = info.ve
            elif method in [
                    QM.CONVEX_HULL, QM.KITCHEN_SINK,
                    QM.KITCHEN_SINK_V2, QM.KITCHEN_SINK_V3,
                    QM.KITCHEN_SINK_V4]:
                pv = info.fp

        elif vs_name == Q_HALLU:
            method = self.q_hallu_method
            if method in [QM.GREEDY, QM.DIVERSE_GREEDY]:
                pv = mi_info[info.pi].ve

        elif vs_name == Q_CHILD:
            method = self.q_child_method
            if method in [QM.GREEDY, QM.DIVERSE_GREEDY]:
                pv = mi_info[info.pi].ve
            elif method == QM.GREEDY_STATS:
                pv = 1. - info.stat[0]

        if pv is None:
            # common methods are after special and we check pv first,
            # so that special case can overwrite common if needed.
            if method == QM.FIFO:
                pv = 1. - 1. / float(1 + model_iter)
            elif method == QM.FILO:
                pv = 1. / float(1 + model_iter)
            elif method == QM.RANDOM:
                pv = np.random.uniform()
            elif method == QM.PREDICT:
                pred_feat = critic_feature(self, mi_info, model_iter, l_info)
                pv = self.predict_one(pred_feat, vs_name)

        assert pv is not None, "pv is None for {} at mi {}".format(vs_name, model_iter)
        queue.add(
            model_str=info.mstr, model_iter=model_iter,
            parent_iter=info.pi, search_depth=info.sd, priority=pv)
        return queue


    def _update_convex_hull_parents(self, q_parent, mi_info):
        # maintain a lower convex hull of finished model
        # select the ones on hull or near hull.
        # probability proportional to cost to the next critical point
        hull_mi = [pqe.model_iter for pqe in q_parent.all_as_generator()]
        hull_info = [mi_info[i] for i in hull_mi]
        xs = [hi.fp for hi in hull_info]
        ys = [hi.ve for hi in hull_info]
        eps = self.q_parent_convex_hull_eps
        allow_increase = self.do_convex_hull_increase
        _hull_indices, eps_indices = _convex_hull_from_points(
            xs, ys, eps=eps, allow_increase=allow_increase)
        # remove all points that are not on the epsilon convex hull
        q_parent.keep_indices_no_auto_update(eps_indices)
        return q_parent

    def _log_convex_hull_parent_choice(self, q_parent, mi_info, e_idx):
        l_pqef = [
            pqef for pqef in q_parent.all_as_generator(full_info=True)
        ]
        l_mi, l_ve, l_fp, l_te, l_cnt = [], [], [], [], []
        for pqef in l_pqef:
            mi = pqef[IDX_PQE].model_iter
            l_mi.append(mi)
            l_ve.append(mi_info[mi].ve)
            l_fp.append(mi_info[mi].fp)
            l_cnt.append(pqef[IDX_CNT])
        logger.info(
            "CONVEX HULL info:\nl_fp={}\nl_ve={}\nl_cnt={}\nl_mi={}".format(
            l_fp, l_ve, l_cnt, l_mi
            ))
        logger.info("Chose parent e_idx={} mi={}".format(e_idx, l_mi[e_idx]))

    def _choose_parent_hull(self, q_parent, mi_info):
        q_parent = self._update_convex_hull_parents(q_parent, mi_info)
        # full information queue entries
        # note that the queue is in the order of computational cost
        # or decreasing in error rate, since they are taken out of the
        # performance convex hull
        l_pqef = [
            pqef for pqef in q_parent.all_as_generator(full_info=True)
        ]
        l_cnts = np.asarray(
            [pqef[IDX_CNT] for pqef in l_pqef],
            dtype=np.float32
        )
        n_xs = len(l_cnts)
        e_idx = -1
        while e_idx < 0:
            for _e_idx in reversed(range(n_xs)):
                cnt = l_cnts[_e_idx]
                rand_val = np.random.uniform()
                if rand_val <= 1.0 / (cnt + 1.0):
                    e_idx = _e_idx
                    break
        self._log_convex_hull_parent_choice(q_parent, mi_info, e_idx)
        return q_parent.peek_at(e_idx)

    def _choose_parent_sink(self, q_parent, mi_info, method):
        """
        Using the kitchen sink of intuitions to guide the choice
        of parents
        """
        # update the parent queue so it has the updated convex hull.
        q_parent = self._update_convex_hull_parents(q_parent, mi_info)
        l_pqef = [
            pqef for pqef in q_parent.all_as_generator(full_info=True)
        ]
        # going through each entry to compute the unormalized weights.
        l_weights = []
        for pqef, npqef in zip(l_pqef, l_pqef[1:] + [None]):
            # Intuition 1: inverse of how often it is sampled
            weight = 1.0
            cnt = np.float32(pqef[IDX_CNT])
            weight /= (cnt + 1.0)

            # val error, computational cost of curr and next
            info = mi_info[pqef[IDX_PQE].model_iter]
            ve, fp = info.ve, info.fp
            if npqef is not None:
                next_info = mi_info[npqef[IDX_PQE].model_iter]
                nve, nfp = next_info.ve, next_info.fp
            else:
                # special case of the last one
                nve = ve
                nfp = self.options.search_max_flops

            # Intuition 2: inverse of computational cost
            weight /= fp

            # Intuition 3: based on distance on x and y
            # based on x: the more difference,
            #   the more it should be sampled
            # based on y: the more difference,
            #   the less it should be sampled
            eps_fp = 1e-3
            eps_ve = 1e-2
            fp_diff = max(nfp - fp, eps_fp * fp)
            ve_diff = max(ve - nve, eps_ve * ve)
            fp_rdiff = fp_diff / fp
            inv_ve_rdiff = ve / ve_diff

            QM = QueueSortMethods
            if method == QM.KITCHEN_SINK:
                weight *= fp_diff
            elif method == QM.KITCHEN_SINK_V2:
                weight *= fp_diff * inv_ve_rdiff
            elif method == QM.KITCHEN_SINK_V3:
                weight *= fp_rdiff
            elif method == QM.KITCHEN_SINK_V4:
                weight *= fp_rdiff * inv_ve_rdiff
            l_weights.append(weight)

        l_weights = np.asarray(l_weights)
        l_weights = np.log(l_weights)
        max_weight = np.max(l_weights)
        exp_weights = np.exp(l_weights - max_weight)
        prob = exp_weights / np.sum(exp_weights)

        e_idx = np.random.choice(
            range(len(l_weights)), p=prob)
        self._log_convex_hull_parent_choice(q_parent, mi_info, e_idx)
        return q_parent.peek_at(e_idx)


    def choose_parent(self, q_parent, mi_info):
        QM = QueueSortMethods
        method = self.q_parent_method
        n_reuses = self.options.n_parent_reuses
        if n_reuses is None or n_reuses < 0:
            # infinity reuse.
            if method == QM.GREEDY:
                return q_parent.peek()
            elif method == QM.RANDOM:
                return q_parent.random_peek()
            elif method == QM.DIVERSE_GREEDY:
                # form a diverse set of parents, and select one.
                if len(self.diverse_parent) == 0:
                    self.diverse_parent = q_parent.diverse_top_k(
                        mi_info, self.div_opts)
                    logger.info("Diverse parents are {}".format(
                        [x.model_iter for x in self.diverse_parent]))
                ret = self.diverse_parent[0]
                self.diverse_parent[0:1] = []
                return ret
            elif method == QM.CONVEX_HULL:
                return self._choose_parent_hull(q_parent, mi_info)
            elif method in [
                    QM.KITCHEN_SINK, QM.KITCHEN_SINK_V2,
                    QM.KITCHEN_SINK_V3, QM.KITCHEN_SINK_V4]:
                return self._choose_parent_sink(q_parent, mi_info, method)
            else:
                raise NotImplementedError("Meow")
        else:
            # limited reuse
            return q_parent.pop()


    def initial_net_info(self):
        l_master, normal_func, reduction_func = self._initial_net_info()
        n_normal_inputs = 0 if normal_func is None else normal_func.n_inputs
        n_reduction_inputs = 0 if reduction_func is None else reduction_func.n_inputs
        l_net_info = []
        cell_based = self.options.search_cell_based
        # macro version does not allow cat end_merge,
        # because the loading is crazily complicated. :-(
        if cell_based and self.options.search_cat_based:
            logger.info("Macro search disallows cat based")
            self.options.search_cat_based = False
        end_merge = (LayerTypes.MERGE_WITH_CAT if self.options.search_cat_based
            else LayerTypes.MERGE_WITH_SUM)

        if cell_based:
            for master in l_master:
                normal = reduction = None
                if normal_func is not None:
                    normal = normal_func(end_merge=end_merge)
                if reduction_func is not None:
                    reduction = reduction_func(end_merge=end_merge)
                l_net_info.append(CellNetworkInfo(master=master,
                    normal=normal, reduction=reduction))
        else:
            for master in l_master:
                layer_id = 0
                cell_outputs = []
                flat_master = LayerInfoList()
                for info in master:
                    if LayerInfo.is_input(info):
                        cell_outputs.append(layer_id)
                        layer_id += 1
                        flat_master.append(info)
                        continue
                    aux_weight = info.aux_weight
                    inputs = [cell_outputs[cell_id] for cell_id in info.inputs]
                    if info.merge_op == 'normal':
                        l_info = normal_func(
                            next_id=layer_id - n_normal_inputs,
                            input_ids=inputs,
                            end_merge=end_merge)[n_normal_inputs:]

                    elif info.merge_op == 'reduction':
                        l_info = reduction_func(
                            next_id=layer_id - n_reduction_inputs,
                            input_ids=inputs,
                            end_merge=end_merge)[n_reduction_inputs:]
                        # TODO need to check this b/c there are multiple downsample
                        l_info[0].down_sampling = 1
                        layer_id = l_info[-1].id + 1
                    else:
                        l_info = [ copy.deepcopy(info) ]
                        l_info[-1].inputs = inputs
                        l_info[-1].id = layer_id

                    cell_outputs.append(l_info[-1].id)
                    layer_id = l_info[-1].id + 1
                    flat_master.extend(l_info)
                    flat_master[-1].aux_weight = aux_weight
                l_net_info.append(CellNetworkInfo(master=flat_master,
                    normal=None, reduction=None))
            #end for
        #end if-else cell_based
        return l_net_info

    def _initial_net_info(self):
        raise NotImplementedError(
            "Derived class will implement this along with ds specific")


class RecognitionController(PetridishController):

    def __init__(self, options):
        super(RecognitionController, self).__init__(options)
        if options.use_hallu_feat_sel:
            merge_op = LayerTypes.MERGE_WITH_WEIGHTED_SUM
        else:
            merge_op = LayerTypes.MERGE_WITH_SUM
        self.merge_operations = [merge_op]
        self.valid_operations = [
            LayerTypes.SEPARABLE_CONV_3_2,
            LayerTypes.SEPARABLE_CONV_5_2,
            LayerTypes.DILATED_CONV_3,
            LayerTypes.DILATED_CONV_5,
            LayerTypes.MAXPOOL_3x3,
            LayerTypes.AVGPOOL_3x3,
            LayerTypes.IDENTITY
        ]

    def _initial_net_info(self):
        options = self.options
        cell_type = self.options.init_cell_type
        if cell_type is None:
            normal_func = separable_resnet_cell_info
            reduction_func = separable_resnet_cell_info
        elif cell_type == 'nasneta':
            normal_func = nasneta_cell_info
            reduction_func = nasnata_reduction_cell_info
        elif cell_type == 'resnet_basic':
            normal_func = basic_resnet_cell_info
            reduction_func = basic_resnet_cell_info

        n_normal_inputs = normal_func.n_inputs
        n_reduction_inputs = reduction_func.n_inputs
        l_num_cells_per_reduce = list(map(int, options.init_n.strip().split(',')))
        l_master = []
        for nc_per_reduce in l_num_cells_per_reduce:
            num_reduction_layers = options.n_reduction_layers
            num_cells = nc_per_reduce * (num_reduction_layers + 1)
            num_init_reductions = 0
            skip_reduction_layer_input = 0

            master = CellNetworkInfo.default_master(
                n_normal_inputs, n_reduction_inputs, num_cells,
                num_reduction_layers, num_init_reductions,
                skip_reduction_layer_input, self.options.use_aux_head)
            l_master.append(master)
        return l_master, normal_func, reduction_func


class MLPController(PetridishController):

    def __init__(self, options):
        super(MLPController, self).__init__(options)
        self.valid_operations = [
            LayerTypes.FC_TANH_MUL_GATE,
            LayerTypes.FC_RELU_MUL_GATE,
            LayerTypes.FC_SGMD_MUL_GATE,
            LayerTypes.FC_IDEN_MUL_GATE
        ]
        if options.use_hallu_feat_sel:
            merge_op = LayerTypes.MERGE_WITH_WEIGHTED_SUM
        else:
            merge_op = LayerTypes.MERGE_WITH_SUM
            # LayerTypes.MERGE_WITH_CAT_PROJ
        self.merge_operations = [merge_op]

    def _initial_net_info(self):
        options = self.options
        cell_type = self.options.init_cell_type
        if cell_type is None:
            normal_func = fully_connected_resnet_cell_info
            reduction_func = None

        n_normal_inputs = normal_func.n_inputs
        n_reduction_inputs = 0
        l_num_cells_per_reduce = list(map(int, options.init_n.strip().split(',')))
        l_master = []
        for nc_per_reduce in l_num_cells_per_reduce:
            num_cells = nc_per_reduce
            num_reduction_layers = 0
            num_init_reductions = 0
            skip_reduction_layer_input = 0

            master = CellNetworkInfo.default_master(
                n_normal_inputs, n_reduction_inputs, num_cells,
                num_reduction_layers, num_init_reductions,
                skip_reduction_layer_input, self.options.use_aux_head)
            l_master.append(master)
        return l_master, normal_func, reduction_func


class RNNController(PetridishController):

    def __init__(self, options):
        super(RNNController, self).__init__(options)
        self.valid_operations = [
            LayerTypes.FC_TANH_MUL_GATE,
            LayerTypes.FC_RELU_MUL_GATE,
            LayerTypes.FC_SGMD_MUL_GATE,
            LayerTypes.FC_IDEN_MUL_GATE
        ]
        if options.use_hallu_feat_sel:
            merge_op = LayerTypes.MERGE_WITH_WEIGHTED_SUM
        else:
            merge_op = LayerTypes.MERGE_WITH_SUM
            # LayerTypes.MERGE_WITH_CAT_PROJ
        self.merge_operations = [merge_op]

    def _initial_net_info(self, cell_based=False, cell_type=None):
        options = self.options
        if cell_type is None:
            normal_func = darts_rnn_base_cell_info
            reduction_func = None

        n_normal_inputs = normal_func.n_inputs
        n_reduction_inputs = 0
        l_num_cells_per_reduce = list(map(int, options.init_n.strip().split(',')))
        l_master = []
        for nc_per_reduce in l_num_cells_per_reduce:
            num_cells = nc_per_reduce
            num_reduction_layers = 0
            num_init_reductions = 0
            skip_reduction_layer_input = 0

            master = CellNetworkInfo.default_master(
                n_normal_inputs, n_reduction_inputs, num_cells,
                num_reduction_layers, num_init_reductions,
                skip_reduction_layer_input, self.options.use_aux_head)
            l_master.append(master)
        return l_master, normal_func, reduction_func


class PetridishRecover(object):

    def __init__(self, controller, options):
        self.controller = controller
        self.n_models_to_recover = options.n_models_to_recover
        self.do_partial_recover = options.do_partial_recover
        if self.n_models_to_recover is None:
            if self.do_partial_recover:
                self.n_models_to_recover = 1

        delattr(options, 'n_models_to_recover')
        delattr(options, 'do_partial_recover')

    @staticmethod
    def add_parser_arguments(parser):
        parser.add_argument('--do_partial_recover', default=False, action='store_true')
        parser.add_argument('--n_models_to_recover', default=None, type=int,
            help="Means different thing with do_partial_recover. If partial recover, it means "+\
                "number of paths to insert into mi_info. This case defaults to 1."+\
                "If full recover, it means the top number of finished models to insert "+\
                "into q_parent. This case defaults to all. ")
        return parser

    def recover(self, *args, **kwargs):
        if not self.do_partial_recover:
            return self.full_recovery(*args, **kwargs)
        else:
            return self.partial_recovery(*args, **kwargs)

    def partial_recovery(self, prev_log_root, log_root, prev_model_root, model_root,
            q_parent, q_hallu, q_child, mi_info):
        """
        deprecated DO NOT USE

        prev_log_root (str) : root of previous log, e.g., on philly: xxx/app_id/logs/2/petridish_main
        log_root (str) : current root of log
        prev_model_root (str) : previous model root, e.g., on philly: xxx/app_id/models/2
        model_root (str) : current model root
        q_parent (PetridishQueue) : see PetridishController.init_queues
        q_hallu (PetridishQueue) :
        q_child (PetridishQueue) :
        mi_info (list) : of ModelSearchInfo
        """
        old_npz = _mi_info_save_fn(prev_log_root)
        old_mi_info = list(np.load(old_npz, encoding='bytes')['mi_info'])

        def _is_hallu(info):
            return info.sd % 2 == 1

        def _is_finished(info):
            return info.ve is not None and info.ve <= 1.0

        min_ve_info = None
        min_ve = None
        for info in old_mi_info:
            if not _is_finished(info):
                continue
            if min_ve is None or (info.ve <= 1.0 and info.ve < min_ve):
                min_ve = info.ve
                min_ve_info = info

        if min_ve is None:
            # nothing finished. start regularly
            return

        # get the path to root.
        info = min_ve_info
        l_info = [info]
        while info.pi != info.mi:
            info = old_mi_info[info.pi]
            l_info.append(info)

        curr_iter = -1
        for info in reversed(l_info):
            curr_iter += 1
            # copy logs into the new dir
            old_log_dir = _mi_to_dn(prev_log_root, info.mi)
            new_log_dir = _mi_to_dn(log_root, curr_iter)
            shutil.copytree(old_log_dir, new_log_dir)
            # copy models
            old_model_dir = _mi_to_dn(prev_model_root, info.mi)
            new_model_dir = _mi_to_dn(model_root, curr_iter)
            shutil.copytree(old_model_dir, new_model_dir)

            info.mi = curr_iter
            info.pi = max(curr_iter - 1, 0)
            mi_info.append(info)

        info = mi_info[-1]
        queue = q_hallu if _is_hallu(info) else q_child
        queue.add(model_str=info.mstr,
                model_iter=info.mi,
                parent_iter=info.pi,
                search_depth=info.sd,
                priority=info.ve)


    def full_recovery(self, prev_log_root, log_root, prev_model_root, model_root,
            q_parent, q_hallu, q_child, mi_info):
        # 0. assume that controller predictor is loaded
        # 1. load old mi_info
        # 2. filter mi_info into models from q_parent, q_hallu, and q_child
        # 3. Copy/link finished model log_dir/model_dir for those mi_info that has ve.
        # 4. add to q_hallu and q_child the ones that are not finished
        # 5. add to q_parent the models that are finished (and are not in hallu)
        # 6. TODO Compute counter variables like n_recv and friends
        def _is_hallu(info):
            return info.sd % 2 == 1

        def _is_finished(info):
            return info.ve is not None and info.ve < 1.0

        prev_mi_info_npz = _mi_info_save_fn(prev_log_root)
        mi_info_npz = _mi_info_save_fn(log_root)
        if os.path.exists(mi_info_npz):
            # Current trial has some mi_info already load from it instead
            # This happens on local runs because they don't have trial id.
            # This also happens on preempt on philly, which doesn't not advance trial id.
            prev_mi_info_npz = mi_info_npz
            prev_model_root = model_root
            prev_log_root = log_root

        if not os.path.exists(prev_mi_info_npz):
            # nothing to load. Return False to let outside know.
            return False

        mi_info.extend(np.load(prev_mi_info_npz, encoding='bytes')['mi_info'])

        if mi_info_npz != prev_mi_info_npz:
            os.rename(prev_mi_info_npz, mi_info_npz)

        all_mi_in_log = set(_all_mi(prev_log_root))
        all_mi_in_model = set(_all_mi(prev_model_root))

        for info in mi_info:
            mi = info.mi
            if mi in all_mi_in_log:
                all_mi_in_log.remove(mi)
            if mi in all_mi_in_model:
                all_mi_in_model.remove(mi)

            # TODO use heapify instaed of inserting one by one...
            old_log_dir = _mi_to_dn(prev_log_root, mi)
            old_model_dir = _mi_to_dn(prev_model_root, mi)
            queue = None
            if not _is_finished(info):
                queue = q_hallu if _is_hallu(info) else q_child
                # remove partial model/logs to avoid confusions.
                if os.path.exists(old_model_dir):
                    shutil.rmtree(old_model_dir)
                if os.path.exists(old_log_dir):
                    shutil.rmtree(old_log_dir)
                logger.info("Recover: mi={} queue={}".format(mi, queue.name))

            else:
                # copy logs
                new_log_dir = _mi_to_dn(log_root, mi)
                if new_log_dir != old_log_dir:
                    shutil.copytree(old_log_dir, new_log_dir)
                # copy models
                new_model_dir = _mi_to_dn(model_root, mi)
                if new_model_dir != old_model_dir:
                    shutil.copytree(old_model_dir, new_model_dir)
                queue = None if _is_hallu(info) else q_parent
                qname = "" if queue is None else queue.name
                # It's important to log val_err for the purpose of analysis later
                logger.info("Recover: mi={} val_err={} queue={}".format(mi, info.ve, qname))

            if queue is not None:
                self.controller.add_one_to_queue(queue, mi_info, mi, None)
        #end for each info

        # remove old mi that have log or model but are not in mi_info
        for mi in all_mi_in_log:
            old_log_dir = _mi_to_dn(prev_log_root, mi)
            if os.path.exists(old_log_dir):
                shutil.rmtree(old_log_dir)

        for mi in all_mi_in_model:
            old_model_dir = _mi_to_dn(prev_model_root, mi)
            if os.path.exists(old_model_dir):
                shutil.rmtree(old_model_dir)

        if not self.n_models_to_recover is None:
            # TODO compute priority and do nsmallest instead of trimming.
            q_parent.keep_top_k(self.n_models_to_recover)
            logger.info("full_recovery: trim q_parent to size {}".format(self.n_models_to_recover))

        # recover successfully
        return True