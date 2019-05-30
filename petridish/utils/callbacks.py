# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf

from tensorpack.callbacks import Callback, HookToCallback, Inferencer

class PerStepHookWithControlDependencies(Callback):

    _chief_only = False

    def __init__(
            self,
            op_func=lambda : 0,
            dependencies_func=lambda self : []):
        """
        Args:
        op_func : op_func() creates an operation
        dependencies_func : dependencies_func(self) finds
            tensors/ops that are in the dependencies.
        """
        self.op_func = op_func
        self.dependencies_func = dependencies_func

    def _setup_graph(self):
        dependencies = self.dependencies_func(self)
        with tf.control_dependencies(dependencies):
            self._fetches = [self.op_func()]

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=self._fetches)


class PerStepInferencer(Inferencer):

    def __init__(
            self,
            op_func=lambda : 0):
        """
        Args:
        op_func : op_func() creates an operation
        """
        self.op_func = op_func

    def get_fetches(self):
        return [self.op_func()]

    def _on_fetches(self, results):
        return