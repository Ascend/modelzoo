from __future__ import print_function
import tensorflow as tf
import numpy as np


class LossValidationHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches=[tf.train.get_global_step(), 'cross_entropy:0'])

    def after_run(self, run_context, run_values):
        global_step, loss = run_values.results
        print("11111111111111")
        print(global_step)
        print(loss)
        if global_step == 1:
            assert (abs(loss - 15.276) < 0.03), "step 0 loss error"
        if global_step == 2:
            assert (abs(loss - 14.313) < 0.05), "step 1 loss error"
