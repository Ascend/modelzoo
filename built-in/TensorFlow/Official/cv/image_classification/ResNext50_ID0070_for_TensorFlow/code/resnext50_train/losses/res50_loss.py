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

class Loss:
    def __init__(self,config):
        self.config = config 

    def get_loss(self, logits, labels):
        labels_one_hot = tf.one_hot(labels, self.config['num_classes'])
        loss = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels_one_hot,label_smoothing=self.config['label_smoothing'])
        loss = tf.identity(loss, name='loss')
        return loss

    def get_total_loss(self, loss):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + reg_losses, name='total_loss')
        return total_loss
 

    def optimize_loss(self, total_loss, opt):
        gate_gradients = (tf.train.Optimizer.GATE_NONE)
        # grads_and_vars = opt.compute_gradients(total_loss, colocate_gradients_with_ops=True, gate_gradients=gate_gradients)
        grads_and_vars = opt.compute_gradients(total_loss, gate_gradients=gate_gradients)

        # train_op = opt.apply_gradients( grads_and_vars, global_step=None )
        train_op = opt.apply_gradients( grads_and_vars)

        return train_op

   


        



