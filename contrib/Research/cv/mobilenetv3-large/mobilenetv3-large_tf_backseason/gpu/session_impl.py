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
import time
import pandas as pd
from datasets import dataset_factory
from nets import nets_factory


class SessionImpl:
	def __init__(self, env):
		self.env = env

	def main(self):
		if not self.env.FLAGS.dataset_dir:
			raise ValueError('You must supply the dataset directory with --dataset_dir')

		tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
		# tf.config.optimizer.set_experimental_options({"arithmetic_optimization": False})
		with tf.Graph().as_default():
			global_step = self.env.slim.create_global_step()
			preprocessing_name = None
			if self.env.FLAGS.augment_images:
				preprocessing_name = self.env.FLAGS.preprocessing_name or self.env.FLAGS.model_name

			# Gather initial summaries.
			summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
			train_op, iterator_train, total_loss = \
				self.create_train_ops(global_step, preprocessing_name, summaries)

			eval_accuracy, metric_update_op, iterator_eval = self.create_eval_ops(preprocessing_name)
			summary_op = tf.summary.merge(list(summaries), name='summary_op')
			##### xling #####
			self.train(train_op, summary_op=summary_op, total_loss=total_loss,
												global_step=global_step, iterator_train=iterator_train,
												eval_accuracy=eval_accuracy,
												metric_update_op=metric_update_op, iterator_eval=iterator_eval)

	##### xling end #####

	def create_train_ops(self, global_step, preprocessing_name, summaries):
		num_classes = 1001
		network_fn = nets_factory.get_network_fn(
			self.env.FLAGS.model_name,
			num_classes=(num_classes - self.env.FLAGS.labels_offset),
			weight_decay=self.env.FLAGS.weight_decay,
			is_training=True)  # is_training is finally passed to mobilenet() of mobilenet.py but is not used.

		######################
		# Select the dataset #
		######################
		dataset_train = dataset_factory.get_dataset(
			self.env.FLAGS.dataset_name, self.env.FLAGS.dataset_split_name, self.env.FLAGS.dataset_dir)

		images_train, labels_train, iterator_train = \
			self.create_data(dataset_train, preprocessing_name, is_training=True)

		####################
		# Define the model #
		####################

		logits_train = self.env.calc_logits(network_fn, images_train)

		loss, total_loss = self.env.calc_loss(logits_train, labels_train)
		summaries.add(tf.compat.v1.summary.scalar('loss', loss))
		train_op = self.env.create_train_op(global_step, summaries, total_loss)
		return train_op, iterator_train, total_loss

	def create_eval_ops(self, preprocessing_name):
		num_classes = 1001
		network_fn = nets_factory.get_network_fn(
			self.env.FLAGS.model_name,
			num_classes=(num_classes - self.env.FLAGS.labels_offset),
			weight_decay=self.env.FLAGS.weight_decay,
			is_training=False)  # is_training is finally passed to mobilenet() of mobilenet.py but is not used.

		dataset_eval = dataset_factory.get_dataset(
			self.env.FLAGS.dataset_name, 'validation', self.env.FLAGS.dataset_dir)
		images_eval, labels_eval, iterator_eval = \
			self.create_data(dataset_eval, preprocessing_name, is_training=False)
		logits_eval = self.env.calc_logits(network_fn, images_eval)
		predictions = tf.argmax(logits_eval, 1)
		eval_accuracy, metric_update_op = tf.compat.v1.metrics.accuracy(tf.argmax(labels_eval, 1), predictions)
		tf.compat.v1.summary.scalar('metric_update_op', metric_update_op)
		return eval_accuracy, metric_update_op, iterator_eval

	def train(self, train_op, summary_op, total_loss, global_step, iterator_train=None,
						eval_accuracy=None, metric_update_op=None, iterator_eval=None):
		saver = tf.compat.v1.train.Saver(max_to_keep=5)
		import time
		t_global_start = time.time()
		t_local_start = t_global_start

		config = tf.compat.v1.ConfigProto()
		if self.env.FLAGS.enable_hvd:
			config.gpu_options.visible_device_list = str(self.env.hvd.local_rank())
			# config.intra_op_parallelism_threads = 1
			# config.inter_op_parallelism_threads = 5

		log_df = pd.DataFrame(columns=['step', 'loss', 'fps_per_gpu', 'time_per_step', 'total_steps'])

		with tf.compat.v1.Session(config=config) as sess:
			sess.run(iterator_train.initializer)
			if self.env.FLAGS.measure_accu_during_train:
				sess.run(iterator_eval.initializer)

			if self.env.FLAGS.checkpoint_path == '':
				sess.run(tf.compat.v1.global_variables_initializer())
				sess.run(tf.compat.v1.local_variables_initializer())
			else:
				if tf.io.gfile.isdir(self.env.FLAGS.checkpoint_path):
					checkpoint_path = tf.train.latest_checkpoint(self.env.FLAGS.checkpoint_path)
				else:
					checkpoint_path = self.env.FLAGS.checkpoint_path
				sess.run(tf.compat.v1.global_variables_initializer())
				sess.run(tf.compat.v1.local_variables_initializer())
				saver.restore(sess, f'{checkpoint_path}')

			if self.env.FLAGS.enable_hvd:
				sess.run(self.env.hvd.broadcast_variables(tf.trainable_variables(), 0))

			logdir = self.env.create_logdir()
			summary_writer = None
			if self.env.FLAGS.enable_hvd:
				if self.env.hvd.rank() == 0:
					summary_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
			else:
				summary_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			step = 0
			for step in range(self.env.FLAGS.max_number_of_steps):
				##### train ######
				t_local_start, _global_step = self._train(sess, train_op, total_loss, global_step, summary_op,
																												 summary_writer, step, t_local_start, log_df, logdir)

				##### evaluate ######
				if self.env.FLAGS.measure_accu_during_train and step % self.env.FLAGS.log_every_n_steps == 0:
					if (self.env.FLAGS.enable_hvd and self.env.hvd.rank() == 0) or (
					not self.env.FLAGS.enable_hvd):
						_metric_update_op = sess.run([metric_update_op])
						print(f'step: {_global_step}, _metric_update_op: {_metric_update_op}')

						acc = sess.run([eval_accuracy])
						print(f'acc: {acc}')

				##### save ckpt ######
				if step % self.env.FLAGS.ckp_freq == 0:
					if self.env.FLAGS.enable_hvd:
						if self.env.hvd.rank() == 0:
							saver.save(sess, "%s/model.ckpt" % logdir, global_step=step)
					else:
						saver.save(sess, "%s/model.ckpt" % logdir, global_step=step)

			perf = (step * self.env.FLAGS.batch_size) / (time.time() - t_global_start)
			print("perf (samples processed per second): %f" % perf)

			coord.request_stop()
			coord.join(threads)

	def _get_init_fn(self):
		if self.env.FLAGS.checkpoint_path is None:
			return None

		# Warn the user if a checkpoint exists in the train_dir. Then we'll be
		# ignoring the checkpoint anyway.
		if tf.train.latest_checkpoint(self.env.FLAGS.train_dir):
			tf.compat.v1.logging.info(
				'Ignoring --checkpoint_path because a checkpoint already exists in %s'
				% self.env.FLAGS.train_dir)
			return None

		exclusions = []
		if self.env.FLAGS.checkpoint_exclude_scopes:
			exclusions = [scope.strip()
										for scope in self.env.FLAGS.checkpoint_exclude_scopes.split(',')]

		# TODO(sguada) variables.filter_variables()
		variables_to_restore = []
		for var in self.env.slim.get_model_variables():
			for exclusion in exclusions:
				if var.op.name.startswith(exclusion):
					break
			else:
				variables_to_restore.append(var)

		if tf.io.gfile.isdir(self.env.FLAGS.checkpoint_path):
			checkpoint_path = tf.train.latest_checkpoint(self.env.FLAGS.checkpoint_path)
		else:
			checkpoint_path = self.env.FLAGS.checkpoint_path

		tf.compat.v1.logging.info('Fine-tuning from %s' % checkpoint_path)

		return self.env.slim.assign_from_checkpoint_fn(
			checkpoint_path,
			variables_to_restore,
			ignore_missing_vars=self.env.FLAGS.ignore_missing_vars)

	def _train(self, sess, train_op, total_loss, global_step, summary_op, summary_writer, step,
						 t_local_start, log_df, logdir):
		if self.env.FLAGS.enable_summary:
			_, _total_loss, _global_step, _summary = sess.run([train_op, total_loss, global_step, summary_op])
			if self.env.FLAGS.enable_hvd:
				if self.env.hvd.rank() == 0:
					summary_writer.add_summary(_summary, step)
			else:
				summary_writer.add_summary(_summary, step)
		else:
			_, _total_loss, _global_step = sess.run([train_op, total_loss, global_step])

		if step % self.env.FLAGS.log_every_n_steps == 0:
			if (self.env.FLAGS.enable_hvd and self.env.hvd.rank() == 0) or (
			not self.env.FLAGS.enable_hvd):
				t_local_end = time.time()
				num_sample_processed = self.env.FLAGS.log_every_n_steps * self.env.FLAGS.batch_size
				fps_per_gpu = num_sample_processed / (t_local_end - t_local_start)
				time_per_step = (t_local_end - t_local_start) / self.env.FLAGS.log_every_n_steps
				print(
					"step: %d/%d, perf per GPU (samples processed per second): %f, time per step: %f, _total_loss: %f, global_step: %d" %
					(_global_step, self.env.FLAGS.max_number_of_steps, fps_per_gpu, time_per_step, _total_loss, _global_step))
				log_df.loc[log_df.shape[0]] = [step, _total_loss, fps_per_gpu, time_per_step,
																			 self.env.FLAGS.max_number_of_steps]
				log_fn = f'{logdir}/log.csv'
				log_df.to_csv(log_fn, index=False)
				t_local_start = t_local_end

		return t_local_start, _global_step

	def create_data(self, dataset, preprocessing_name, is_training):
		if not self.env.FLAGS.fake_input:
			#### true data #####
			from dataloader import data_provider
			iterator, _ = data_provider.get_data(dataset, self.env.FLAGS.batch_size,
																					 dataset.num_classes, self.env.FLAGS.labels_offset, is_training,
																					 preprocessing_name, self.env.FLAGS.use_grayscale,
																					 self.env.hvd, self.env.FLAGS.enable_hvd)
			images, labels = iterator.get_next()
			images = tf.reshape(images, [self.env.FLAGS.batch_size, 224, 224, 3])  # (32, 224, 224, 3), float32
			labels = tf.reshape(labels, [self.env.FLAGS.batch_size, dataset.num_classes])  # (32, 1001), float32
			tf.compat.v1.summary.image('input images', images, max_outputs=4)

		else:
			iterator = None
			##### fake data #####
			images = tf.constant(2, dtype=tf.float32, shape=[self.env.FLAGS.batch_size, 224, 224, 3])
			labels = tf.constant(2, dtype=tf.float32, shape=[self.env.FLAGS.batch_size, dataset.num_classes])
		#### xling end #####

		return images, labels, iterator
