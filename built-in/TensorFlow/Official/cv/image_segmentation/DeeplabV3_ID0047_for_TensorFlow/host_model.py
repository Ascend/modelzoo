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
import model
import six
import common
from utils import train_utils
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib.framework import get_variables_to_restore, assign_from_checkpoint_fn
import numpy as np
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer


_NUM_EXAMPLES_NAME="num_examples"
slim = tf.contrib.slim


FLAGS = tf.app.flags.FLAGS

class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)


class Model(object):
    def __init__(self):

        self.outputs_to_num_classes = {'semantic':32}
        self.ignore_label = 255
        self.model_options = common.ModelOptions(
            outputs_to_num_classes=self.outputs_to_num_classes,
            crop_size=[int(sz) for sz in FLAGS.train_crop_size],
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

    def get_estimator_model_func(self, features, labels, mode, params=None):

        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.train_fn(features,labels,mode)
        else:
            return self.evaluate_fn(features,labels,mode)


    def train_fn(self,features,labels,mode):

        samples = {}
        outputs_to_num_classes = self.outputs_to_num_classes
        ignore_label = self.ignore_label
        model_options = self.model_options

        samples[common.IMAGE] = features
        samples[common.LABEL] = labels

        outputs_to_scales_to_logits = model.multi_scale_logits(
            samples[common.IMAGE],
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid,
            weight_decay=FLAGS.weight_decay,
            is_training=True,
            fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
            nas_training_hyper_parameters={
                'drop_path_keep_prob': FLAGS.drop_path_keep_prob,
                'total_training_steps': FLAGS.training_number_of_steps,
            })

        # Add name to graph node so we can add to summary.
        output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
        output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
            output_type_dict[model.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE)

        for output, num_classes in six.iteritems(outputs_to_num_classes):
            train_utils.add_softmax_cross_entropy_loss_for_each_scale(
                outputs_to_scales_to_logits[output],
                samples[common.LABEL],
                num_classes,
                ignore_label,
                loss_weight=model_options.label_weights,
                upsample_logits=FLAGS.upsample_logits,
                hard_example_mining_step=FLAGS.hard_example_mining_step,
                top_k_percent_pixels=FLAGS.top_k_percent_pixels,
                scope=output)

        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        reg = tf.add_n(regularization_losses)
        reg_loss = tf.identity(reg, "reg_loss")

        loss = tf.get_collection(tf.GraphKeys.LOSSES)
        loss = tf.add_n(loss)
        loss = tf.identity(loss, name="loss")

        all_losses = []
        all_losses.append(loss)
        all_losses.append(reg_loss)
        total_loss = tf.add_n(all_losses)

        total_loss = tf.identity(total_loss, name='total_loss')

        global_step = tf.train.get_global_step()

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy,
            FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step,
            FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps,
            FLAGS.learning_power,
            FLAGS.slow_start_step,
            FLAGS.slow_start_learning_rate,
            decay_steps=FLAGS.decay_steps,
            end_learning_rate=FLAGS.end_learning_rate)

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

        opt = NPUDistributedOptimizer(optimizer)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []

        gate_gradients = tf.train.Optimizer.GATE_NONE
        grads_and_vars = opt.compute_gradients(total_loss, gate_gradients=gate_gradients)

        last_layers = model.get_extra_layer_scopes(
            FLAGS.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(
            last_layers, FLAGS.last_layer_gradient_multiplier, FLAGS.bias_multiplier)
        if grad_mult:
            grads_and_vars = slim.learning.multiply_gradients(
                grads_and_vars, grad_mult)

        # Create gradient update op.
        grad_updates = opt.apply_gradients(
            grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')


        var_list = get_variables_to_restore(exclude=['global_step'])
        init_fn = assign_from_checkpoint_fn(FLAGS.tf_initial_checkpoint, var_list, ignore_missing_vars=True)

        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op,
                                          training_hooks=[RestoreHook(init_fn)])

    def evaluate_fn(self,features,labels,mode):

        samples = {}

        ignore_label = self.ignore_label
        model_options = self.model_options

        samples[common.IMAGE] = features
        samples[common.LABEL] = labels

        samples[common.IMAGE].set_shape(
            [FLAGS.eval_batch_size,
             int(FLAGS.eval_crop_size[0]),
             int(FLAGS.eval_crop_size[1]),
             3])
        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')
            predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                               image_pyramid=FLAGS.image_pyramid)
        else:
            tf.logging.info('Performing multi-scale test.')
            if FLAGS.quantize_delay_step >= 0:
                raise ValueError(
                    'Quantize mode is not supported with multi-scale test.')

            predictions = model.predict_labels_multi_scale(
                samples[common.IMAGE],
                model_options=model_options,
                eval_scales=FLAGS.eval_scales,
                add_flipped_images=FLAGS.add_flipped_images)
        predictions = predictions[common.OUTPUT_TYPE]
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(samples[common.LABEL], shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, ignore_label))
        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights = 0.

        labels = tf.where(
            tf.equal(labels, ignore_label), tf.zeros_like(labels), labels)

        predictions_tag = 'miou'
        for eval_scale in FLAGS.eval_scales:
            predictions_tag += '_' + str(eval_scale)
        if FLAGS.add_flipped_images:
            predictions_tag += '_flipped'

        # Define the evaluation metric.
        metric_map = {}
        num_classes = 21

        # IoU for each class.
        one_hot_predictions = tf.one_hot(predictions, num_classes)
        one_hot_predictions = tf.reshape(one_hot_predictions, [-1, num_classes])
        one_hot_labels = tf.one_hot(labels, num_classes)
        one_hot_labels = tf.reshape(one_hot_labels, [-1, num_classes])
        for c in range(num_classes):
            predictions_tag_c = '%s_class_%d' % (predictions_tag, c)
            tp, tp_op = tf.metrics.true_positives(
                labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
                weights=weights)
            fp, fp_op = tf.metrics.false_positives(
                labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
                weights=weights)
            fn, fn_op = tf.metrics.false_negatives(
                labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
                weights=weights)
            tp_fp_fn_op = tf.group(tp_op, fp_op, fn_op)
            iou = tf.where(tf.greater(tp + fn, 0.0),
                           tp / (tp + fn + fp),
                           tf.constant(np.NaN))
            metric_map['eval/%s' % predictions_tag_c] = (iou, tp_fp_fn_op)

        total_loss = tf.losses.get_total_loss()
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metric_map)
