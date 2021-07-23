# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from time import gmtime, strftime
from tensorflow.contrib import slim as contrib_slim
from gpu_helper import get_custom_getter
import random
import numpy as np
import os

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)


class Env:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.slim = contrib_slim
        self.num_samples = 1281167


    def _configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.

        Args:
            learning_rate: A scalar or `Tensor` learning rate.

        Returns:
            An instance of an optimizer.

        Raises:
            ValueError: if Initializer.FLAGS.optimizer is not recognized.
        """
        if self.FLAGS.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self.FLAGS.adadelta_rho,
                epsilon=self.FLAGS.opt_epsilon)
        elif self.FLAGS.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=self.FLAGS.adagrad_initial_accumulator_value)
        elif self.FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=self.FLAGS.adam_beta1,
                beta2=self.FLAGS.adam_beta2,
                epsilon=self.FLAGS.opt_epsilon)
        elif self.FLAGS.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self.FLAGS.ftrl_learning_rate_power,
                initial_accumulator_value=self.FLAGS.ftrl_initial_accumulator_value,
                l1_regularization_strength=self.FLAGS.ftrl_l1,
                l2_regularization_strength=self.FLAGS.ftrl_l2)
        elif self.FLAGS.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self.FLAGS.momentum,
                name='Momentum')
        elif self.FLAGS.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self.FLAGS.rmsprop_decay,
                momentum=self.FLAGS.rmsprop_momentum,
                epsilon=self.FLAGS.opt_epsilon)
        elif self.FLAGS.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized' % self.FLAGS.optimizer)

        return optimizer


    def create_logdir(self):
        logdir = self.FLAGS.checkpoint_path + "/results"
        os.makedirs(logdir, exist_ok=True)
        return logdir


    def calc_logits(self, network_fn, images):
        logits, end_points = network_fn(images, reuse=tf.AUTO_REUSE)
        return logits


    def calc_loss(self, logits_train, labels_train):
        base_loss = self.slim.losses.softmax_cross_entropy(
            logits_train, labels_train, label_smoothing=self.FLAGS.label_smoothing, weights=1.0)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([base_loss] + reg_losses, name='total_loss')

        loss = tf.add_n([base_loss])
        loss = tf.identity(loss, name='loss')

        return loss, total_loss


    def calc_steps_per_epoch(self):
        return self.num_samples // (self.FLAGS.batch_size * int(os.getenv('RANK_SIZE')))
        

    def _configure_learning_rate(self, global_step):
        steps_per_epoch = self.calc_steps_per_epoch()
        decay_steps = int(steps_per_epoch * self.FLAGS.num_epochs_per_decay)

        if self.FLAGS.learning_rate_decay_type == 'exponential':
            learning_rate = tf.train.exponential_decay(
                self.FLAGS.learning_rate,
                global_step,
                decay_steps,
                self.FLAGS.learning_rate_decay_factor,
                staircase=True,
                name='exponential_decay_learning_rate')
        elif self.FLAGS.learning_rate_decay_type == 'fixed':
            learning_rate = tf.constant(self.FLAGS.learning_rate, name='fixed_learning_rate')
        elif self.FLAGS.learning_rate_decay_type == 'cosine_annealing':
            current_step_epoch = global_step//steps_per_epoch *steps_per_epoch
            learning_rate = tf.train.cosine_decay(self.FLAGS.learning_rate, current_step_epoch, self.FLAGS.max_number_of_steps)
        elif self.FLAGS.learning_rate_decay_type == 'polynomial':
            learning_rate = tf.train.polynomial_decay(
                self.FLAGS.learning_rate, global_step,
                decay_steps,
                self.FLAGS.end_learning_rate,
                power=1.0,
                cycle=False,
                name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                                             self.FLAGS.learning_rate_decay_type)

        if self.FLAGS.warmup_epochs:
            warmup_lr = (
                self.FLAGS.learning_rate * tf.cast(global_step, tf.float32) /
                (steps_per_epoch * self.FLAGS.warmup_epochs))
            learning_rate = tf.minimum(warmup_lr, learning_rate)

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        # tf.Print(learning_rate, [learning_rate], '*****************')
        return learning_rate


    def create_train_op(self, global_step, summaries, loss):
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []

        #################################
        # Configure the moving averages #
        #################################
        if self.FLAGS.moving_average_decay:
            moving_average_variables = self.slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                self.FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        learning_rate = self._configure_learning_rate(global_step)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if self.FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))


        opt = self._configure_optimizer(learning_rate)

        from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
        from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
        from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
        from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
        loss_scale_manager = FixedLossScaleManager(loss_scale=4096)
        #loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=1024, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        if int(os.getenv('RANK_SIZE')) == 1:
            opt = NPULossScaleOptimizer(opt, loss_scale_manager)
        else:
            opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=True)
        opt = NPUDistributedOptimizer(opt)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            gate_gradients = (tf.train.Optimizer.GATE_NONE)
            grads_and_vars = opt.compute_gradients(loss)
            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        return train_op
