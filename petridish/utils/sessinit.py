import os
import numpy as np
import tensorflow as tf
import six

from tensorpack.utils import logger
from tensorpack.tfutils.common import (
    get_op_tensor_name, get_global_step_var)
from tensorpack.tfutils.varmanip import SessionUpdate
from tensorpack.tfutils.sessinit import (
    SessionInit, SaverRestore, CheckpointReaderAdapter)

__all__ = ['SaverRestoreSizeRelaxed', 'read_parameter_val']


class SaverRestoreSizeRelaxed(SaverRestore):
    """ Same as :class:`SaverRestore`, but has more relaxed constraints.

        It allows loading variable of difference sizes, but of the same number of dimensions.
        The lower of value of each dim is the chosen dimension value.
        The first chunk of the each dim of the value is loaded into the variable.
    """
    def _run_init(self, sess):
        logger.info(
            "Restoring checkpoint with size relaxation from {} ...".format(self.path))

        def f(reader, name, v):
            val = reader.get_tensor(name)
            val_shape = list(val.shape)
            var_shape = v.get_shape().as_list()
            if var_shape != val_shape:
                n_dims = len(val_shape)
                assert len(var_shape) == n_dims, \
                    "Size Relaxation requires the variable match in number of dimensions"
                slices = []
                pad_params = []
                logger.info(
                    "Loading variable {} with var_shape {} and val_shape {}".format(
                        name, var_shape, val_shape))
                for var_s, val_s in zip(var_shape, val_shape):
                    if var_s > val_s:
                        pad_params.append([0, var_s - val_s])
                    else:
                        pad_params.append([0, 0])
                    slices.append(slice(0, var_s))
                val = np.pad(val, pad_params, 'constant')[slices]
            SessionUpdate.load_value_to_var(v, val)
        with sess.as_default():
            self._match_vars(f)

class AssignGlobalStep(SessionInit):

    def __init__(self, global_step_val):
        self.global_step_val = global_step_val
        self.assign_op = None

    def _setup_graph(self):
        global_step = get_global_step_var()
        self.assign_op = global_step.assign(self.global_step_val)

    def _run_init(self, sess):
        sess.run(self.assign_op)


def read_parameter_val(model_dir, l_names):
    model_path = tf.train.latest_checkpoint(model_dir)
    reader = tf.train.NewCheckpointReader(model_path)
    reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
    return [ reader.get_tensor(var_name) for var_name in l_names ]