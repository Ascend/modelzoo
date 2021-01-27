# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
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

import tensorflow as tf
from time import gmtime, strftime
from tensorflow.contrib import slim as contrib_slim
from gpu_helper import get_custom_getter
import random
import numpy as np
import os

np.random.seed(0)
random.seed(0)
tf.compat.v1.set_random_seed(0)


class Env:
	def __init__(self, FLAGS, hvd):
		self.FLAGS = FLAGS

		self.slim = contrib_slim
		self.num_samples = 1281167
		self.optimizer_config = {
			'loss_scale': 'Backoff',
			'scale_min': 1.0,
			'scale_max': 1024.0,
			'step_window': 200,
			'iter_size': 1.0,
			'use_lars': False,
			'weight_decay': self.FLAGS.weight_decay,
			'dtype': tf.float16 if self.FLAGS.enable_mixed_precision else tf.float32
		}

		self.hvd = hvd


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
			optimizer = tf.compat.v1.train.MomentumOptimizer(
				learning_rate,
				momentum=self.FLAGS.momentum,
				name='Momentum')
		elif self.FLAGS.optimizer == 'rmsprop':
			optimizer = tf.compat.v1.train.RMSPropOptimizer(
				learning_rate,
				decay=self.FLAGS.rmsprop_decay,
				momentum=self.FLAGS.rmsprop_momentum,
				epsilon=self.FLAGS.opt_epsilon)
		elif self.FLAGS.optimizer == 'sgd':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		else:
			raise ValueError('Optimizer [%s] was not recognized' % self.FLAGS.optimizer)

		#  if Initializer.FLAGS.enable_hvd:
		#    optimizer = hvd.DistributedOptimizer(optimizer)
		#
		#  optimizer = MixedPrecisionOptimizer(optimizer, scale=1024)
		return optimizer


	def create_logdir(self):
		logdir = "results/%s" % strftime("%m%d%H%M%S_train", gmtime())
		logdir += f'_hvd{self.FLAGS.enable_hvd}' \
			f'_mn{self.FLAGS.model_name}' \
			f'_augmented{self.FLAGS.augment_images}' \
			f'_mixedp{self.FLAGS.enable_mixed_precision}' \
			f'_lr{self.FLAGS.learning_rate}' \
			f'_opt{self.FLAGS.optimizer}' \
			f'_me{self.FLAGS.max_epoch}' \
			f'_lrdt{self.FLAGS.learning_rate_decay_type}' \
			f'_nepd{self.FLAGS.num_epochs_per_decay}' \
			f'_lrdf{self.FLAGS.learning_rate_decay_factor}' \
			f'_rm{self.FLAGS.run_mode}' \
			f'_dlm{self.FLAGS.data_loader_mode}' \
			f'_b{self.FLAGS.batch_size}'
		logdir += f'_{self.FLAGS.msg}'
		logdir = "results"
		#logdir = "results/0615083319_train_hvdTrue_mnmobilenet_v2_augmentedTrue_mixedpFalse_lr0.1_optmomentum_me150_lrdtcosine_annealing_nepd0.3125_lrdf0.98_rmestimator_dlmunited_b256_me_param"

		from shutil import copyfile
		if (self.FLAGS.enable_hvd and self.hvd.rank() == 0) or (not self.FLAGS.enable_hvd):
			pass
			# os.makedirs(logdir)
	#		copyfile('./scripts/run_hvd_official.sh', f'{logdir}/run_hvd_official.sh')
	#		copyfile('./scripts/run_hvd_me.sh', f'{logdir}/run_hvd_me.sh')
	#		copyfile('./scripts/run_hvd_torch.sh', f'{logdir}/run_hvd_torch.sh')
		return logdir


	def calc_logits(self, network_fn, images):
		if self.FLAGS.enable_mixed_precision:
			images = tf.cast(images, tf.float16)
			with tf.variable_scope("mobilenet_v2", custom_getter=get_custom_getter(tf.float16)):
				logits, end_points = network_fn(images, reuse=tf.compat.v1.AUTO_REUSE)
			logits = tf.cast(logits, tf.float32)
		else:
			#logits, end_points = network_fn(images)
			logits, end_points = network_fn(images, reuse=tf.compat.v1.AUTO_REUSE)

		return logits


	def calc_loss(self, logits_train, labels_train):
		base_loss = self.slim.losses.softmax_cross_entropy(
			logits_train, labels_train, label_smoothing=self.FLAGS.label_smoothing, weights=1.0)

		reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
		total_loss = tf.add_n([tf.saturate_cast(base_loss, self.optimizer_config['dtype'])] + reg_losses,
													name='total_loss')

		loss = tf.add_n([base_loss])
		loss = tf.identity(loss, name='loss')

		return loss, total_loss


	def calc_steps_per_epoch(self):
		if self.FLAGS.enable_hvd:
			return self.num_samples // (self.FLAGS.batch_size * self.hvd.size())
		#return self.num_samples // (self.FLAGS.batch_size)
		return self.num_samples // (self.FLAGS.batch_size * int(os.getenv('RANK_SIZE')))


	def _configure_learning_rate(self, global_step):
		steps_per_epoch = self.calc_steps_per_epoch()
		decay_steps = int(steps_per_epoch * self.FLAGS.num_epochs_per_decay)

		if self.FLAGS.learning_rate_decay_type == 'exponential':
			learning_rate = tf.compat.v1.train.exponential_decay(
				self.FLAGS.learning_rate,
				global_step,
				decay_steps,
				self.FLAGS.learning_rate_decay_factor,
				staircase=True,
				name='exponential_decay_learning_rate')
		elif self.FLAGS.learning_rate_decay_type == 'fixed':
			learning_rate = tf.constant(self.FLAGS.learning_rate, name='fixed_learning_rate')
		elif self.FLAGS.learning_rate_decay_type == 'cosine_annealing':
			learning_rate = tf.compat.v1.train.cosine_decay(self.FLAGS.learning_rate, global_step,
																						self.FLAGS.max_number_of_steps)

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
		update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) or []

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
		summaries.add(tf.compat.v1.summary.scalar('learning_rate', learning_rate))

		if self.FLAGS.moving_average_decay:
			# Update ops executed locally by trainer.
			update_ops.append(variable_averages.apply(moving_average_variables))

		if self.FLAGS.enable_hvd:
			opt = self._configure_optimizer(learning_rate)
			opt = self.hvd.DistributedOptimizer(opt)

		#  optimizer = op.Optimizer(optimizer_config)
		#  opt = optimizer.get_lbs_optimizer(_configure_optimizer(learning_rate),
		#                                    optimizer_config['iter_size'])
		else:
			opt = self._configure_optimizer(self.FLAGS.learning_rate)

		#from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
		#opt = NPUDistributedOptimizer(opt)

		update_op = tf.group(*update_ops)
		with tf.control_dependencies([update_op]):
			gate_gradients = (tf.compat.v1.train.Optimizer.GATE_NONE)
			# grads_and_vars = opt.compute_gradients(loss, colocate_gradients_with_ops=True, gate_gradients=gate_gradients)
			grads_and_vars = opt.compute_gradients(loss)
			if self.FLAGS.enable_hvd:
				# train_op = opt.apply_gradients(grads_and_vars, loss_scale=1024)
				train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
			else:
				train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

		return train_op
