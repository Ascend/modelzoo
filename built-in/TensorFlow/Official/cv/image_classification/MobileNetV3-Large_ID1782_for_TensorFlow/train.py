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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from env import Env

from estimator_impl import EstimatorImpl

import os


def parse_args():
    tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
    tf.app.flags.DEFINE_string('train_dir', './mobilenet_v1_result',
                                                         'Directory where checkpoints and event logs are written to.')
    tf.app.flags.DEFINE_string('msg', '', 'extra message for creating log folder')
    tf.app.flags.DEFINE_string('gpu_ids', '0', 'the gpu to use')

    # cosine learning rate. [DECOUPLED WEIGHT DECAY REGULARIZATION]
    tf.app.flags.DEFINE_string('checkpoint_path', '', '')
    tf.app.flags.DEFINE_integer('max_epoch', None, 'max epochs to train')
    tf.app.flags.DEFINE_integer('max_train_steps', None, 'max steps to train')
    tf.app.flags.DEFINE_float('eta_min', '0.0', 'eta_min in cosine_annealing scheduler')
    tf.app.flags.DEFINE_integer('T_max', '200', 'T-max in cosine_annealing scheduler')
    tf.app.flags.DEFINE_integer('ckp_freq', '5000', 'Frequency (in steps) to save checkpoint')
    tf.app.flags.DEFINE_integer('iterations_per_loop', None, 'Iterations per loop when running on Ascend')

    tf.app.flags.DEFINE_float('warmup_epochs', 5, 'Linearly warmup learning rate from 0 to learning_rate over this many epochs.')
    tf.app.flags.DEFINE_boolean('enable_summary', False, '')

    tf.app.flags.DEFINE_integer('num_clones', 1,
        'Number of model clones to deploy. Note For '
        'historical reasons loss from all clones averaged '
        'out and learning rate decay happen per clone '
        'epochs')

    tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                                                            'Use CPUs to deploy clones.')

    tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

    tf.app.flags.DEFINE_integer(
        'num_ps_tasks', 0,
        'The number of parameter servers. If the value is 0, then the parameters '
        'are handled locally by the worker.')

    tf.app.flags.DEFINE_integer(
        'num_readers', 4,
        'The number of parallel readers that read data from the dataset.')

    tf.app.flags.DEFINE_integer(
        'num_preprocessing_threads', 4,
        'The number of threads used to create the batches.')

    tf.app.flags.DEFINE_integer(
        'log_every_n_steps', 10,
        'The frequency with which logs are print.')

    tf.app.flags.DEFINE_integer(
        'save_summaries_secs', 60,
        'The frequency with which summaries are saved, in seconds.')

    tf.app.flags.DEFINE_integer(
        'save_interval_secs', 60,
        'The frequency with which the model is saved, in seconds.')

    tf.app.flags.DEFINE_integer(
        'task', 0, 'Task id of the replica running the training.')

    ######################
    # Optimization Flags #
    ######################

    tf.app.flags.DEFINE_float(
        # 'weight_decay', 0.00004, 'The weight decay on the model weights.')
        'weight_decay', 0, 'The weight decay on the model weights.')

    tf.app.flags.DEFINE_string(
        'optimizer', 'sgd',
        'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
        '"ftrl", "momentum", "sgd" or "rmsprop".')

    tf.app.flags.DEFINE_float(
        'adadelta_rho', 0.95,
        'The decay rate for adadelta.')

    tf.app.flags.DEFINE_float(
        'adagrad_initial_accumulator_value', 0.1,
        'Starting value for the AdaGrad accumulators.')

    tf.app.flags.DEFINE_float(
        'adam_beta1', 0.9,
        'The exponential decay rate for the 1st moment estimates.')

    tf.app.flags.DEFINE_float(
        'adam_beta2', 0.999,
        'The exponential decay rate for the 2nd moment estimates.')

    tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

    tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                                                        'The learning rate power.')

    tf.app.flags.DEFINE_float(
        'ftrl_initial_accumulator_value', 0.1,
        'Starting value for the FTRL accumulators.')

    tf.app.flags.DEFINE_float(
        'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

    tf.app.flags.DEFINE_float(
        'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

    tf.app.flags.DEFINE_float(
        'momentum', 0.9,
        'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

    tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

    tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

    tf.app.flags.DEFINE_integer(
        'quantize_delay', -1,
        'Number of steps to start quantized training. Set to -1 would disable '
        'quantized training.')

    #######################
    # Learning Rate Flags #
    #######################

    tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'fixed',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

    # tf.app.flags.DEFINE_float('learning_rate', 0.4, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

    tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.0001,
        'The minimal end learning rate used by a polynomial decay learning rate.')

    tf.app.flags.DEFINE_float(
        'label_smoothing', 0.1, 'The amount of label smoothing.')

    tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

    tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays. Note: this flag counts '
        'epochs per clone but aggregates per sync replicas. So 1.0 means that '
        'each clone will go over full epoch individually, but replicas will go '
        'once across all replicas.')

    tf.app.flags.DEFINE_bool(
        'sync_replicas', False,
        'Whether or not to synchronize the replicas during training.')

    tf.app.flags.DEFINE_integer(
        'replicas_to_aggregate', 1,
        'The Number of gradients to collect before updating params.')

    tf.app.flags.DEFINE_float(
        'moving_average_decay', None,
        'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.')

    #######################
    # Dataset Flags #
    #######################

    tf.app.flags.DEFINE_string(
        'dataset_name', 'imagenet', 'The name of the dataset to load.')

    tf.app.flags.DEFINE_string(
        'dataset_split_name', 'train', 'The name of the train/test split.')

    tf.app.flags.DEFINE_string(
        'dataset_dir', '/opt/npu/models/slimImagenet', 'The directory where the dataset files are stored.')

    tf.app.flags.DEFINE_integer(
        'labels_offset', 0,
        'An offset for the labels in the dataset. This flag is primarily used to '
        'evaluate the VGG and ResNet architectures which do not use a background '
        'class for the ImageNet dataset.')

    tf.app.flags.DEFINE_string(
        'model_name', 'mobilenet_v1', 'The name of the architecture to train.')

    tf.app.flags.DEFINE_string(
        'preprocessing_name', 'inception_v2', 'The name of the preprocessing to use. If left '
                                                                                    'as `None`, then the model_name flag is used.')

    tf.app.flags.DEFINE_integer(
        'batch_size', 96, 'The number of samples in each batch.')

    tf.app.flags.DEFINE_integer(
        'train_image_size', None, 'Train image size')

    tf.app.flags.DEFINE_integer('max_number_of_steps', 20000,
                                                            'The maximum number of training steps.')

    tf.app.flags.DEFINE_bool('use_grayscale', False,
                                                     'Whether to convert input images to grayscale.')

    #####################
    # Fine-Tuning Flags #
    #####################
    #
    # tf.app.flags.DEFINE_string(
    #   'checkpoint_path', None,
    #   'The path to a checkpoint from which to fine-tune.')
    #
    tf.app.flags.DEFINE_string(
        'checkpoint_exclude_scopes', None,
        'Comma-separated list of scopes of variables to exclude when restoring '
        'from a checkpoint.')

    tf.app.flags.DEFINE_string(
        'trainable_scopes', None,
        'Comma-separated list of scopes to filter the set of variables to train.'
        'By default, None would train all the variables.')

    tf.app.flags.DEFINE_boolean(
        'ignore_missing_vars', False,
        'When restoring a checkpoint would ignore missing variables.')

    FLAGS = tf.app.flags.FLAGS

    return FLAGS


FLAGS = parse_args()


def main(_):

    num_samples = 1281167

    if FLAGS.max_train_steps is None and FLAGS.max_epoch is None:
        raise ValueError("Must set max_train_steps or max_epoch.")

    if FLAGS.max_train_steps is not None and FLAGS.max_epoch is not None:
        raise ValueError("Can't set both max_train_steps and max_epoch.")

    if FLAGS.max_train_steps:
        FLAGS.max_number_of_steps = FLAGS.max_train_steps
    else:
        FLAGS.max_number_of_steps = num_samples // (FLAGS.batch_size * int(os.getenv('RANK_SIZE'))) * FLAGS.max_epoch

    env = Env(FLAGS)

    estimator_impl = EstimatorImpl(env)
    estimator_impl.main()


if __name__ == '__main__':
    tf.app.run()
