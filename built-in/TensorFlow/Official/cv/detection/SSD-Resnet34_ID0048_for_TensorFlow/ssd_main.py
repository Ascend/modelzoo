# Copyright 2018 Google. All Rights Reserved.
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
"""Training script for SSD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import multiprocessing
import os

import sys
import threading
from absl import app
import numpy as np
import tensorflow as tf

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf import rewriter_config_pb2
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator  import NPUEstimator

import coco_metric
import dataloader
import ssd_constants
import ssd_model

def get_rank_size():
    return int(os.environ['RANK_SIZE'])

tf.flags.DEFINE_string('model_dir', None, 'Location of model_dir')
tf.flags.DEFINE_string('resnet_checkpoint', '',
                       'Location of the ResNet checkpoint to use for model '
                       'initialization.')
tf.flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
tf.flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
tf.flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                                              'evaluation.')
tf.flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
tf.flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
tf.flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
tf.flags.DEFINE_integer('num_examples_per_epoch', 120000,
                        'Number of examples in one epoch')
tf.flags.DEFINE_float('num_epochs', 58, 'Number of epochs for training')

tf.flags.DEFINE_string('mode', 'train_and_eval',
                       'Mode to run: train_and_eval, train, eval')

tf.flags.DEFINE_integer(
    'keep_checkpoint_max', 32,
    'Maximum number of checkpoints to keep.')


FLAGS = tf.flags.FLAGS

SUCCESS = False


def construct_run_config():
    """Construct the run config."""

    # Parse hparams
    hparams = ssd_model.default_hparams()

    params = dict(
        hparams.values(),
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        resnet_checkpoint=FLAGS.resnet_checkpoint,
        val_json_file=FLAGS.val_json_file,
        mode=FLAGS.mode,
        model_dir=FLAGS.model_dir,
        eval_samples=FLAGS.eval_samples,
    )

    return NPURunConfig(
        hcom_parallel=True,
        model_dir=FLAGS.model_dir,
        session_config=tf.ConfigProto(),
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_checkpoints_steps=ssd_constants.CHECKPOINT_FREQUENCY,
        enable_data_pre_proc=True,
        save_summary_steps=100,
        iterations_per_loop=100,
        precision_mode='allow_mix_precision'
      ), params

def coco_eval(predictions,
              current_step,
              summary_writer,
              coco_gt,
              use_cpp_extension=True,
              nms_on_tpu=True):
    """Call the coco library to get the eval metrics."""
    global SUCCESS
    eval_results = coco_metric.compute_map(
        predictions,
        coco_gt,
        use_cpp_extension=use_cpp_extension,
        nms_on_tpu=nms_on_tpu)
    if eval_results['COCO/AP'] >= ssd_constants.EVAL_TARGET and not SUCCESS:
        SUCCESS = True
    tf.logging.info('Eval results: %s' % eval_results)
    # Write out eval results for the checkpoint.
    with tf.Graph().as_default():
        summaries = []
        for metric in eval_results:
            summaries.append(
                tf.Summary.Value(tag=metric, simple_value=eval_results[metric]))
        tf_summary = tf.Summary(value=list(summaries))
        summary_writer.add_summary(tf_summary, current_step)

def init_npu():
   """Initialize npu manually.
   Returns:
     `init_sess` npu  init session config.
     `npu_init` npu  init ops.
   """
   npu_init = npu_ops.initialize_system()
   config = tf.ConfigProto()

   #npu mix precision attribute set to true when using mix precision
   config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
   custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
   custom_op.name = "NpuOptimizer"
   custom_op.parameter_map["use_off_line"].b = True
   custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
   init_sess = tf.Session(config=config)
   return init_sess,npu_init

def main(argv):
    init_sess, npu_init = init_npu()
    init_sess.run(npu_init)

    del argv  # Unused.
    global SUCCESS

    # Check data path
    if FLAGS.mode in ('train',
                      'train_and_eval') and FLAGS.training_file_pattern is None:
        raise RuntimeError('You must specify --training_file_pattern for training.')
    if FLAGS.mode in ('train_and_eval', 'eval'):
        if FLAGS.validation_file_pattern is None:
            raise RuntimeError('You must specify --validation_file_pattern '
                               'for evaluation.')
        if FLAGS.val_json_file is None:
            raise RuntimeError('You must specify --val_json_file for evaluation.')

    run_config, params = construct_run_config()

    if FLAGS.mode == 'train':
        train_params = dict(params)
        train_params['batch_size'] = FLAGS.train_batch_size
        train_estimator = NPUEstimator(
            model_fn=ssd_model.ssd_model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config,
            params=train_params)

        tf.logging.info(params)

        train_estimator.train(
            input_fn=dataloader.SSDInputReader(
                FLAGS.training_file_pattern,
                params['transpose_input'],
                is_training=True),
            steps=int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                      FLAGS.train_batch_size / get_rank_size()))

    elif FLAGS.mode == 'train_and_eval':
        output_dir = os.path.join(FLAGS.model_dir, 'eval')
        tf.gfile.MakeDirs(output_dir)
        # Summary writer writes out eval metrics.
        summary_writer = tf.summary.FileWriter(output_dir)

        current_step = 0

        coco_gt = coco_metric.create_coco(
            FLAGS.val_json_file, use_cpp_extension=params['use_cocoeval_cc'])
        for eval_step in ssd_constants.EVAL_STEPS:
            # Compute the actual eval steps based on the actural train_batch_size
            steps = int(eval_step / get_rank_size() * ssd_constants.DEFAULT_BATCH_SIZE /
                        FLAGS.train_batch_size)
            print('###################################', steps)

            tf.logging.info('Starting training cycle for %d steps.' % steps)
            run_config, params = construct_run_config()

            train_params = dict(params)
            train_params['batch_size'] = FLAGS.train_batch_size
            train_estimator = NPUEstimator(
                model_fn=ssd_model.ssd_model_fn,
                model_dir=FLAGS.model_dir,
                config=run_config,
                params=train_params)
            tf.logging.info(params)
            train_estimator.train(
                input_fn=dataloader.SSDInputReader(
                    FLAGS.training_file_pattern,
                    params['transpose_input'],
                    is_training=True),
                steps=steps)

            if SUCCESS:
                break

            current_step = current_step + steps

            tf.logging.info('Starting evaluation cycle at step %d.' % current_step)
            # Run evaluation at the given step.
            eval_params = dict(params)
            eval_params['batch_size'] = FLAGS.eval_batch_size
            eval_estimator = NPUEstimator(
                model_fn=ssd_model.ssd_model_fn,
                model_dir=FLAGS.model_dir,
                config=run_config,
                params=eval_params)

            predictions = list(
                eval_estimator.predict(
                    input_fn=dataloader.SSDInputReader(
                        FLAGS.validation_file_pattern,
                        is_training=False)))

            coco_eval(predictions, current_step, summary_writer, coco_gt, params['use_cocoeval_cc'], False)
        summary_writer.close()

    elif FLAGS.mode == 'eval':
        coco_gt = coco_metric.create_coco(
            FLAGS.val_json_file, use_cpp_extension=params['use_cocoeval_cc'])
        eval_params = dict(params)
        eval_params['batch_size'] = FLAGS.eval_batch_size
        eval_estimator = NPUEstimator(
            model_fn=ssd_model.ssd_model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config,
            params=eval_params)

        output_dir = os.path.join(FLAGS.model_dir, 'eval')
        tf.gfile.MakeDirs(output_dir)
        # Summary writer writes out eval metrics.
        summary_writer = tf.summary.FileWriter(output_dir)
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.logging.info('Starting to evaluate on newest checkpoint.')
        predictions = list(
            eval_estimator.predict(
                checkpoint_path=ckpt,
                input_fn=dataloader.SSDInputReader(
                    FLAGS.validation_file_pattern,
                    is_training=False)))
        tf.logging.info('Starting to cal coco ap.')
        current_step = int(os.path.basename(ckpt).split('-')[1])

        coco_eval(predictions, current_step, summary_writer, coco_gt,
                  params['use_cocoeval_cc'], False)
        tf.logging.info('end to evaluate.')

        summary_writer.close()

    npu_shutdown = npu_ops.shutdown_system()
    init_sess.run(npu_shutdown)
    init_sess.close()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
