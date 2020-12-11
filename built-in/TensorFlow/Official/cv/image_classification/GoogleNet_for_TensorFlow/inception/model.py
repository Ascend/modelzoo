# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
from . import inception_v1
from . import inception_v4
from tensorflow.contrib import slim as slim
import numpy as np

class Model(object):
    def __init__(self, args, data, hyper_param, layers, logger):
        self.args = args
        self.data = data
        self.hyper_param = hyper_param
        self.layers = layers
        self.logger = logger  

    def get_estimator_model_func(self, features, labels, mode, params=None):
        labels = tf.reshape(labels, (-1,))
    
        inputs = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        inputs = tf.cast(inputs, self.args.dtype)

        if is_training:
            if self.args.network == "inception_v1":
                with slim.arg_scope(inception_v1.inception_v1_arg_scope(weight_decay = self.args.weight_decay)):
                    top_layer, end_points = inception_v1.inception_v1(inputs = features, num_classes = 1000, dropout_keep_prob = 0.7, is_training = True)
            if self.args.network == "inception_v4":
                with slim.arg_scope(inception_v4.inception_v4_arg_scope(weight_decay=self.args.weight_decay)):
                    top_layer, end_points = inception_v4.inception_v4(inputs=features, num_classes=1000, dropout_keep_prob=0.8, is_training = True)
        else:
            if self.args.network == "inception_v1":
                with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
                    top_layer, end_points = inception_v1.inception_v1(inputs = features, num_classes = 1000, dropout_keep_prob = 1.0, is_training = False)
            if self.args.network == "inception_v4":
                with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
                    top_layer, end_points = inception_v4.inception_v4(inputs=features, num_classes=1000, dropout_keep_prob=1.0, is_training = False)

        logits = top_layer
        predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
        logits = tf.cast(logits, tf.float32)

        labels_one_hot = tf.one_hot(labels, depth=1000)

        loss = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels_one_hot, label_smoothing=self.args.label_smoothing)

        base_loss = tf.identity(loss, name='loss')

        l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        l2_loss = tf.multiply(l2_loss, self.args.weight_decay)
        total_loss = base_loss + l2_loss

        # loss = tf.losses.softmax_cross_entropy(logits, labels_one_hot, label_smoothing=self.args.label_smoothing)
        # loss = tf.identity(loss, name='loss')
        # total_loss = tf.losses.get_total_loss(add_regularization_losses = True)

        total_loss = tf.identity(total_loss, name = 'total_loss')

        if mode == tf.estimator.ModeKeys.EVAL:
            with tf.device(None):
                metrics = self.layers.get_accuracy( labels, predicted_classes, logits, self.args)

            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        assert (mode == tf.estimator.ModeKeys.TRAIN)

        batch_size = tf.shape(inputs)[0]

        global_step = tf.train.get_global_step()
        learning_rate = self.hyper_param.get_learning_rate()

        momentum = self.args.momentum

        opt = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=self.args.use_nesterov)

        from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
        opt = NPUDistributedOptimizer(opt)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []

        with tf.control_dependencies(update_ops):
            gate_gradients = tf.train.Optimizer.GATE_NONE
            grads_and_vars = opt.compute_gradients(total_loss, gate_gradients=gate_gradients)
            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        train_op = tf.group(train_op)
        
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)  

