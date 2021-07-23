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

from npu_bridge.npu_init import *
#from npu_bridge import *
import tensorflow as tf
from npu_bridge.estimator import npu_ops
@tf.custom_gradient
def gather_npu(params, indices):
  def grad(dy):
    params_shape = tf.shape(params, out_type=tf.int64)
    params_shape = tf.cast(params_shape, tf.int32)
    grad_gather = tf.unsorted_segment_sum(dy, indices, params_shape[0])
    return grad_gather, None
  return tf.gather(params, indices), grad

class TCNNConfig(object):
    'CNN配置参数'
    embedding_dim = 64
    seq_length = 600
    num_classes = 10  # 类别数
    num_filters = 1024 # 卷积核数目
    kernel_size = 5
    vocab_size = 5000
    hidden_dim = 1024 # 全连接层神经元
    dropout_keep_prob = 0.5
    learning_rate = 0.001
    batch_size = 512  # 每批训练大小
    num_epochs = 10
    print_per_batch = 100
    save_per_batch = 10
    npu_loss_scale = 1

class TextCNN(object):
    '文本分类，CNN模型'

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        'CNN模型'
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = gather_npu(embedding, self.input_x)
        with tf.name_scope('cnn'):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        with tf.name_scope('score'):
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = npu_ops.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            #For NPU
            #self.optim = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)).minimize(self.loss)
            opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            if self.config.npu_loss_scale < 1:
                # disable npu_loss_scale
                opt = NPUDistributedOptimizer(opt)
            """#使用如下Loss Scale 方法，导致精度溢出，先注释掉如下行数
            else:
                # enable npu_dynamic_loss_scale
                if self.config.npu_loss_scale == 1:
                    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32,
                                                                           incr_every_n_steps=1000,
                                                                           decr_every_n_nan_or_inf=2, decr_ratio=0.5)
                # enable npu_static_loss_scale
                elif self.config.npu_loss_scale > 1:
                    loss_scale_manager = FixedLossScaleManager(loss_scale=self.config.npu_loss_scale)

                if int(os.getenv('RANK_SIZE')) == 1:
                    opt = NPULossScaleOptimizer(opt, loss_scale_manager)
                else:
                    opt = NPUDistributedOptimizer(opt)
                    opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=True)
            """
            self.optim = opt.minimize(self.loss)
            # for NPU
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
