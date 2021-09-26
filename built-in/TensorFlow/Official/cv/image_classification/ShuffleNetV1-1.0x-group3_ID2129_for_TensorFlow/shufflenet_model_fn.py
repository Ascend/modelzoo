#
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
#
from npu_bridge.npu_init import *
import tensorflow as tf
import os
from functools import reduce
from operator import mul
from shufflenet import shufflenet


MOMENTUM = 0.9
USE_NESTEROV = True
MOVING_AVERAGE_DECAY = 0.995
global INIT_FLAG
INIT_FLAG=False

class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """
    global INIT_FLAG
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = shufflenet(
        features['images'],num_classes=params['num_classes'],
        is_training=is_training
        #depth_multiplier=params['depth_multiplier']
    )
    predictions = {
        'probabilities': tf.nn.softmax(logits, axis=1),
        'classes': tf.argmax(logits, axis=1, output_type=tf.int32)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )
    init_fn=None
    if INIT_FLAG:
        exclude =['global_step:0']#,'classifier/biases:0','classifier/weights:0'] # ['ShuffleNetV2/Conv1/weights:0','global_step:0','classifier/biases:0','classifier/weights:0'
            #]
        all_variables = tf.contrib.slim.get_variables_to_restore()
        varialbes_to_use=[]
        num_params=0
        for v in all_variables:
            shape=v.get_shape()
            num_params+= reduce(mul, [dim.value for dim in shape], 1)
            if v.name not in exclude:
                varialbes_to_use.append(v)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(tf.train.latest_checkpoint('models/jiu-huan_0.33_9967'), varialbes_to_use, ignore_missing_vars=True)
        print('***********************params: ', num_params)

    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    with tf.name_scope('cross_entropy'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels['labels'], logits=logits)
        loss = tf.reduce_mean(losses, axis=0)
        tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('cross_entropy_loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels['labels'], predictions['classes']),
            'top5_accuracy': tf.metrics.mean(tf.to_float(tf.nn.in_top_k(
                predictions=predictions['probabilities'], targets=labels['labels'], k=5
            )))
        }
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'], global_step,
            params['decay_steps'], params['end_learning_rate'],
            power=1.0
        )  # linear decay
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ### NPU add loss scale
    loss_scale_manager = FixedLossScaleManager(loss_scale=1024)
    
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=USE_NESTEROV)
        ### NPU add loss scale
        optimizer = NPULossScaleOptimizer(optimizer,loss_scale_manager)
        
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.name_scope('evaluation_ops'):
        train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(
            labels['labels'], predictions['classes']
        )), axis=0)
        train_top5_accuracy = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(
            predictions=predictions['probabilities'], targets=labels['labels'], k=5
        )), axis=0)
    tf.summary.scalar('train_accuracy', train_accuracy)
    tf.summary.scalar('train_top5_accuracy', train_top5_accuracy)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=0)
        train_op = ema.apply(tf.trainable_variables())
    if INIT_FLAG:
        INIT_FLAG=False
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, training_hooks=[RestoreHook(init_fn)])
    else:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
        super(RestoreMovingAverageHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        variables_to_restore = ema.variables_to_restore()
        self.load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self.model_dir), variables_to_restore
        )

    def after_create_session(self, sess, coord):
        tf.logging.info('Loading EMA weights...')
        self.load_ema(sess)

