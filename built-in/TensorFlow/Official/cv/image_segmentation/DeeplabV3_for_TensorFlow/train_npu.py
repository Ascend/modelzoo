# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================

import tensorflow as tf
import data_loader as dl
import host_model as ml
from utils import logger as lg
import trainer as tr
from utils import create_session as cs


tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags

flags.DEFINE_string('train_logdir', 'resnet_101',
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_integer('save_checkpoints_steps',500,
                     'How often, in steps, we save the model to disk.')

flags.DEFINE_integer('max_to_save', 5,
                     'max number of checkpoints to save .')


flags.DEFINE_boolean(
    'save_summaries_images', False,
    'Save sample inputs, labels, and semantic predictions as '
    'images to summary.')

# Settings for profiling.

flags.DEFINE_string('profile_logdir', None,
                    'Where the profile files are stored.')

# Settings for training strategy.

flags.DEFINE_enum('optimizer', 'momentum', ['momentum', 'adam'],
                  'Which optimizer to use.')


# Momentum optimizer flags

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step','cosine'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('decay_steps', 0.0,
                   'Decay steps for polynomial learning rate schedule.')

flags.DEFINE_float('end_learning_rate', 0.0,
                   'End learning rate for polynomial learning rate schedule.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# Adam optimizer flags
flags.DEFINE_float('adam_learning_rate', 0.001,
                   'Learning rate for the adam optimizer.')
flags.DEFINE_float('adam_epsilon', 1e-08, 'Adam optimizer epsilon.')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_list('train_crop_size', '513,513',
                  'Image crop size [height, width] during training.')

flags.DEFINE_float('bias_multiplier',2.0,
                   'bias gradients multiplier' )

flags.DEFINE_float(
    'last_layer_gradient_multiplier', 1.0,
    'The gradient multiplier for last layers, which is used to '
    'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Hyper-parameters for NAS training strategy.

flags.DEFINE_float(
    'drop_path_keep_prob', 1.0,
    'Probability to keep each path in the NAS cell when training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Hard example mining related flags.
flags.DEFINE_integer(
    'hard_example_mining_step', 0,
    'The training step in which exact hard example mining kicks off. Note we '
    'gradually reduce the mining percent to the specified '
    'top_k_percent_pixels. For example, if hard_example_mining_step=100K and '
    'top_k_percent_pixels=0.25, then mining percent will gradually reduce from '
    '100% to 25% until 100K steps after which we only mine top 25% pixels.')

flags.DEFINE_float(
    'top_k_percent_pixels', 1.0,
    'The top k percent pixels (in terms of the loss values) used to compute '
    'loss during training. This is useful for hard pixel mining.')

# Quantization setting.
flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


# for evaluation
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('eval_crop_size', '513,513',
                  'Image crop size [height, width] for evaluation.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


# added for NPU
# flags.DEFINE_string('mode', 'train', 'train or evaluate ')

flags.DEFINE_boolean('debug', True, 'debug mode.')

flags.DEFINE_boolean('shard', False, 'use shard or not .')

flags.DEFINE_boolean('do_checkpoint', True, 
                     ' a flag to indicate if save checkpoint files .')

flags.DEFINE_string('mode', 'train', 
                    'train or evaluate ')

flags.DEFINE_string('log_name', 'training.log', 
                   'name of log file ')

flags.DEFINE_integer('display_every', 100, 
                    'log frequency ')

# flags.DEFINE_string('log_dir', 'resnet_101', ' ')

flags.DEFINE_string('checkpoint_dir', None, 
                    'checkpoint for evaluation')

flags.DEFINE_integer('rank', 1, 
                     'number of devices to run')

flags.DEFINE_integer('num_training_samples', 10582, 
                     'training_samples')

flags.DEFINE_integer('num_evaluating_samples', 4952, 
                     'evaluation samples ')

flags.DEFINE_integer('iterations_per_loop', 10, '')




FLAGS = flags.FLAGS

def main():

    data = dl.DataLoader()
    logger = lg.LogSessionRunHook()
    model = ml.Model()
    trainer = tr.Trainer( data, model, logger)
    if FLAGS.mode=='train':
        trainer.train()
    else:
        trainer.evaluate()


if __name__ == '__main__':
    main()

