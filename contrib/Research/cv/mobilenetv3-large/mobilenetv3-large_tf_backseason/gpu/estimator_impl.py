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
from dataloader import data_provider
from datasets import dataset_factory
from nets import nets_factory
import os


class EstimatorImpl:
	def __init__(self, env):
		self.env = env
		# import layers as ly
		# self.layers = ly.Layers()

	def model_fn(self, features, labels, mode, params):
		num_classes = 1001

		summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

		if mode == tf.estimator.ModeKeys.TRAIN:
			network_fn = nets_factory.get_network_fn(
				self.env.FLAGS.model_name,
				num_classes=(num_classes - self.env.FLAGS.labels_offset),
				weight_decay=self.env.FLAGS.weight_decay,
				is_training=True)

			logits = self.env.calc_logits(network_fn, features)
			loss, total_loss = self.env.calc_loss(logits, labels)

			#### accuracy ####
			predictions = tf.argmax(logits, 1)
			accuracy_ops = tf.compat.v1.metrics.accuracy(tf.argmax(labels, 1), predictions)
			tf.identity(accuracy_ops[1], name='train_accuracy')
			#### accuracy ####

			tf.identity(total_loss, 'train_loss')

			global_step = tf.compat.v1.train.get_or_create_global_step()
			# train_op = self.env.create_train_op(global_step, summaries, loss)
			train_op = self.env.create_train_op(global_step, summaries, total_loss)

			estimator_spec = tf.estimator.EstimatorSpec(
				mode=tf.estimator.ModeKeys.TRAIN, loss=total_loss, train_op=train_op)

		elif mode == tf.estimator.ModeKeys.EVAL:
			network_fn = nets_factory.get_network_fn(
				self.env.FLAGS.model_name,
				num_classes=(num_classes - self.env.FLAGS.labels_offset),
				weight_decay=self.env.FLAGS.weight_decay,
				is_training=False)

			logits = self.env.calc_logits(network_fn, features)
			loss, total_loss = self.env.calc_loss(logits, labels)
			predictions = tf.argmax(logits, 1)
			accuracy_ops = tf.compat.v1.metrics.accuracy(tf.argmax(labels, 1), predictions)
			
			# predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
			# accuracy_ops = self.layers.get_accuracy(tf.argmax(labels, 1), predicted_classes, logits)
			tf.identity(accuracy_ops[1], name='eval_accuracy')
			# tf.identity(accuracy_ops['val-top1acc'], name='eval_accuracy')
			# estimator_spec = tf.estimator.EstimatorSpec(
			# 	mode=tf.estimator.ModeKeys.EVAL,
			# 	loss=loss, eval_metric_ops={'accuracy': accuracy_ops})
			estimator_spec = tf.estimator.EstimatorSpec(
				mode=tf.estimator.ModeKeys.EVAL,
				# loss=total_loss, eval_metric_ops=accuracy_ops)
				loss=total_loss, eval_metric_ops={'accuracy': accuracy_ops})

		return estimator_spec

	def main(self):
		logdir = self.env.create_logdir()

		from logger import LogSessionRunHook

		config = {
			'num_training_samples': self.env.num_samples,
			# 'display_every': self.env.FLAGS.log_every_n_steps,
			# for 1p, just per loop print, for 8p, print each epoch
			'display_every': 625,
			#'display_every': self.env.FLAGS.iterations_per_loop if self.env.FLAGS.iterations_per_loop is not None else self.env.calc_steps_per_epoch(),
			'log_name': 'train_log',
			'log_dir': logdir,
			'global_batch_size': self.env.FLAGS.batch_size * self.env.hvd.size() if self.env.FLAGS.enable_hvd else self.env.FLAGS.batch_size,
			#'global_batch_size': self.env.FLAGS.batch_size * int(os.getenv('RANK_SIZE')),
			#'iterations_per_loop': self.env.FLAGS.iterations_per_loop if self.env.FLAGS.iterations_per_loop is not None else self.env.calc_steps_per_epoch()
			'iterations_per_loop': 1
		}

		hooks = [
			# LogSessionRunHook(config, warmup_steps=self.env.FLAGS.warmup_epochs * self.env.calc_steps_per_epoch()),
			tf.estimator.LoggingTensorHook(['train_accuracy', 'train_loss'], every_n_iter=self.env.FLAGS.log_every_n_steps)
		]
		if self.env.FLAGS.enable_hvd:
			hooks.append(self.env.hvd.BroadcastGlobalVariablesHook(0))
		# assert len(hooks) == 2

		estimator_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
		estimator_config.gpu_options.allow_growth = True
		estimator_config.gpu_options.visible_device_list = str(self.env.hvd.local_rank()) \
			if self.env.FLAGS.enable_hvd else '0'
		estimator_config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
		estimator_config.inter_op_parallelism_threads = 5

		print("XLA is activated.")
		estimator_config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

		if (self.env.FLAGS.enable_hvd and self.env.hvd.rank() == 0) or (not self.env.FLAGS.enable_hvd):
			ckpt_freq = self.env.FLAGS.log_every_n_steps
		else:
			ckpt_freq = None
		classifier = tf.estimator.Estimator(
			model_fn=self.model_fn,
			model_dir=logdir,
			params={'num_samples': self.env.num_samples},
			config = tf.estimator.RunConfig(
				session_config=estimator_config,
				save_summary_steps=self.env.FLAGS.log_every_n_steps,
				save_checkpoints_steps=ckpt_freq,
				keep_checkpoint_max=10
			)
		)

		# #################################################################
		# from npu_bridge.estimator.npu.npu_config import NPURunConfig
		# from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

		# self.estimator_config = tf.compat.v1.ConfigProto(
  #       	inter_op_parallelism_threads=10,
  #       	intra_op_parallelism_threads=10,
  #       	allow_soft_placement=True)

		# self.estimator_config.gpu_options.allow_growth = True

		# gpu_thread_count = 2
		# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
		# os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
		# os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
		# os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

		# run_config = NPURunConfig(
		# 	hcom_parallel=True,
		# 	precision_mode="allow_mix_precision",
		# 	enable_data_pre_proc=True,
		# 	#save_checkpoints_steps=112590,
		# 	save_checkpoints_steps=self.env.calc_steps_per_epoch(),
		# 	session_config=self.estimator_config,
		# 	model_dir=logdir,
		# 	# iterations_per_loop=self.env.calc_steps_per_epoch(),
		# 	#iterations_per_loop=self.env.FLAGS.log_every_n_steps,
		# 	iterations_per_loop=config['iterations_per_loop'],
		# 	keep_checkpoint_max=5)

		# classifier =NPUEstimator(
		# 	model_fn= self.model_fn, 
		# 	config= run_config
		# 	)
		# ###################################################################

		classifier.train(
			input_fn=self.train_data,
			#max_steps=500,
			max_steps=self.env.FLAGS.max_number_of_steps,
			hooks=hooks,
		)
		# #epochs_between_evals=50
		# # #self.env.FLAGS.max_epoch=1
		# #for i in range(self.env.FLAGS.max_epoch):
		# # 	classifier.train(
		# # 		input_fn=self.train_data,
		# # 		steps=self.env.calc_steps_per_epoch(),
		# ## 		#max_steps=10,
		# #  		hooks=hooks
		# #  	)
		# # 	if (i+1)%epochs_between_evals==0:
		# #  	    classifier.evaluate(input_fn=self.eval_data)

	def train_data(self):
		dataset = dataset_factory.get_dataset(self.env.FLAGS.dataset_name, 'train', self.env.FLAGS.dataset_dir)

		preprocessing_name = self.env.FLAGS.preprocessing_name or self.env.FLAGS.model_name
		_, ds = data_provider.get_data(dataset, self.env.FLAGS.batch_size,
																	 dataset.num_classes, self.env.FLAGS.labels_offset, True,
																	 preprocessing_name, self.env.FLAGS.use_grayscale,
																	 self.env.hvd, self.env.FLAGS.enable_hvd,
																	 data_loader_mode=self.env.FLAGS.data_loader_mode)

		return ds

	def eval_data(self):
		dataset = dataset_factory.get_dataset(self.env.FLAGS.dataset_name, 'validation', self.env.FLAGS.dataset_dir)

		preprocessing_name = self.env.FLAGS.preprocessing_name or self.env.FLAGS.model_name
		_, ds = data_provider.get_data(dataset, self.env.FLAGS.batch_size,
																	 dataset.num_classes, self.env.FLAGS.labels_offset, False,
																	 preprocessing_name, self.env.FLAGS.use_grayscale,
																	 self.env.hvd, self.env.FLAGS.enable_hvd)
		return ds
