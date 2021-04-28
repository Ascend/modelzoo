# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_factory
from nets import nets_factory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

slim = contrib_slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', ckpt_path,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', ckpt_path, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/data/Datasets/imagenet_TF', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v1', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    from dataloader import data_provider
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    iterator, _ = data_provider.get_data(dataset, FLAGS.batch_size,
                                         dataset.num_classes, FLAGS.labels_offset, is_training=False,
                                         preprocessing_name=preprocessing_name,
                                         use_grayscale=FLAGS.use_grayscale,
                                         hvd=None, enable_hvd=None)
    images, labels = iterator.get_next()  # label: [100, 1001]
    images = tf.reshape(images, [FLAGS.batch_size, 224, 224, 3])  # (100, 224, 224, 3), float32
    labels = tf.argmax(labels, axis=1)  # [100]
    logits, _ = network_fn(images)

    if FLAGS.quantize:
      contrib_quantize.create_eval_graph()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    eval_accuracy, metric_update_op = tf.metrics.accuracy(labels, predictions)

    # tf.summary.scalar('top1_acc', top1_accu)
    # summaries_op = tf.summary.merge_all()

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    ##### evaluate #####
    tf.logging.info('Evaluating %s' % checkpoint_path)
    saver = tf.train.Saver()
    from time import gmtime, strftime
    logdir = "results/%s" % strftime("%m%d%H%M%S_evel", gmtime())
    # summary_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
    with tf.Session() as sess:
      sess.run(iterator.initializer)
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      saver.restore(sess, f'{checkpoint_path}')
      tf.train.write_graph(sess.graph, logdir, 'graph.pbtxt')

      for step in range(num_batches):
        _metric_update_op = sess.run([metric_update_op])
        print(f'{step}, _metric_update_op: {_metric_update_op}')

      acc = sess.run([eval_accuracy])
      print(f'acc: {acc}')


if __name__ == '__main__':
  tf.app.run()
