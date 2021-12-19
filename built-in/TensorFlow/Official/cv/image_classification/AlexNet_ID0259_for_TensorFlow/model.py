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
import tensorflow as tf
from npu_bridge.npu_init import *

class AlexNetModel:
    def __init__(self, input_size, decaying_factor=0.0005, LRN_depth=5, LRN_bias=2, LRN_alpha=0.0001, LRN_beta=0.75):
        self.input_size = input_size
        self.decaying_factor = decaying_factor
        self.LRN_depth = LRN_depth
        self.LRN_bias = LRN_bias
        self.LRN_alpha = LRN_alpha
        self.LRN_beta = LRN_beta

    def classifier(self, img, keep_prob):
        ###### 1st conv_gpu1
        self.W1_1 = tf.get_variable('conv1_gpu1', shape=[11, 11, 3, 48],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B1_1 = tf.get_variable('bias1_gpu1', shape=[48],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
        self.L1_1 = self.conv(img, self.W1_1, self.B1_1,
                              conv_stride=(1, 4, 4, 1), conv_padding='VALID', LRN=True,
                              pooling=True)   # [None, 27, 27, 48]

        ###### 1st conv_gpu2
        self.W1_2 = tf.get_variable('conv1_gpu2', shape=[11, 11, 3, 48],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B1_2 = tf.get_variable('bias1_gpu2', shape=[48],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
        self.L1_2 = self.conv(img, self.W1_2, self.B1_2,
                              conv_stride=(1, 4, 4, 1), conv_padding='VALID', LRN=True,
                              pooling=True)  # [None, 27, 27, 48]

        ###### 2st conv gpu1
        self.W2_1 = tf.get_variable('conv2_gpu1', shape=[5, 5, 48, 128],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B2_1 = tf.get_variable('bias2_gpu1', shape=[128],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L2_1 = self.conv(self.L1_1, self.W2_1, self.B2_1,
                              conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=True,
                              pooling=True)  # [None, 13, 13, 128]

        ###### 2st conv gpu2
        self.W2_2 = tf.get_variable('conv2_gpu2', shape=[5, 5, 48, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B2_2 = tf.get_variable('bias2_gpu2', shape=[128],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L2_2 = self.conv(self.L1_2, self.W2_2, self.B2_2,
                              conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=True,
                              pooling=True)  # [None, 13, 13, 128]

        ###### concat 2 gpu way before 3st conv
        self.L2 = tf.concat([self.L2_1, self.L2_2], axis=3)  # [None, 13, 13, 256]

        ##### 3st conv
        self.W3 = tf.get_variable('conv3', shape=[3, 3, 256, 384],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B3 = tf.get_variable('bias3', shape=[384],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
        self.L3 = self.conv(self.L2, self.W3, self.B3,
                            conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=False,
                            pooling=False)  # [None, 13, 13, 384]

        ##### split into 2 way before 4st conv
        self.L3_1, self.L3_2 = tf.split(self.L3, num_or_size_splits=2, axis=3)  # [None, 13, 13, 192]

        ##### 4st conv gpu1
        self.W4_1 = tf.get_variable('conv4_gpu1', shape=[3, 3, 192, 192],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B4_1 = tf.get_variable('bias4_gpu1', shape=[192],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))

        self.L4_1 = self.conv(self.L3_1, self.W4_1, self.B4_1,
                              conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=False,
                              pooling=False)  # [None, 13, 13, 192]

        ##### 4st conv gpu2
        self.W4_2 = tf.get_variable('conv4_gpu2', shape=[3, 3, 192, 192],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B4_2 = tf.get_variable('bias4_gpu2', shape=[192],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L4_2 = self.conv(self.L3_2, self.W4_2, self.B4_2,
                              conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=False,
                              pooling=False)  # [None, 13, 13, 192]

        ##### 5st conv gpu1
        self.W5_1 = tf.get_variable('conv5_gpu1', shape=[3, 3, 192, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B5_1 = tf.get_variable('bias5_gpu1', shape=[128],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L5_1 = self.conv(self.L4_1, self.W5_1, self.B5_1,
                              conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=False,
                              pooling=True)  # [None, 6, 6, 128]

        ##### 5st conv gpu2
        self.W5_2 = tf.get_variable('conv5_gpu2', shape=[3, 3, 192, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.B5_2 = tf.get_variable('bias5_gpu2', shape=[128],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L5_2 = self.conv(self.L4_2, self.W5_2, self.B5_2,
                              conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=False,
                              pooling=True)  # [None, 6, 6, 128]

        ###### concat 2 gpu way before 1st fcl
        self.L5 = tf.concat([self.L5_1, self.L5_2], axis=3)  # [None, 6, 6, 256]
        ###### and flatten
        self.L5 = tf.reshape(self.L5, shape=[-1, 6 * 6 * 256])  # [None, 6 * 6 * 256]

        ##### 1st fc
        self.W6 = tf.get_variable('fc1', shape=[6 * 6 * 256, 4096],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L6 = tf.matmul(self.L5, self.W6)  # [None, 4096]
        self.B6 = tf.get_variable('bias6', shape=[4096],
                                  dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L6 = tf.nn.bias_add(self.L6, self.B6)
        # self.L6 = tf.nn.relu(tf.nn.dropout(self.L6, keep_prob=keep_prob))
        self.L6 = tf.nn.relu(npu_ops.dropout(self.L6, keep_prob=keep_prob))

        ##### 2st fc
        ##### this is original #####
        ##self.W7 = tf.get_variable('fc2', shape=[4096, 4096],
        ##                          dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        ##self.B7 = tf.get_variable('bias7', shape=[4096],
        ##                          dtype = tf.float32, initializer = tf.constant_initializer(1))
        ############################
        self.W7 = tf.get_variable('fc2', shape=[4096, 1000],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L7 = tf.matmul(self.L6, self.W7)  # [None, 1000]
        self.B7 = tf.get_variable('bias7', shape=[1000],
                                  dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.L7 = tf.nn.bias_add(self.L7, self.B7)
        # self.L7 = tf.nn.relu(tf.nn.dropout(self.L7, keep_prob=keep_prob))
        self.L7 = tf.nn.relu(npu_ops.dropout(self.L7, keep_prob=keep_prob))

        ##### 3st fc
        ##### this is original #####
        ##self.W8 = tf.get_variable('fc3', shape=[4096, 2],
        ##                          dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        ## self.B8 = tf.get_variable('bias8', shape=[2],
        ##                          dtype=tf.float32, initializer=tf.constant_initializer(1))
        ############################
        self.W8 = tf.get_variable('fc3', shape=[1000, 2],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L8 = tf.matmul(self.L7, self.W8)  # [None, 2]
        self.B8 = tf.get_variable('bias8', shape=[2],
                                  dtype=tf.float32, initializer=tf.constant_initializer(1))
        self.logit = tf.nn.bias_add(self.L8, self.B8)


        ##### collect weights for weight decay
        ##### weight decay is the methods for giving panalty to too-big weights which is likely to induce co-adaptations.
        ##### bias do not need to decay, because they all are added equaly with each of input node's channel.
        ##### so they are not likely to induce co-adaptation
        ##### effect of decaying biases will be trivial
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W4_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W4_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W5_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W5_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W6), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W7), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W8), self.decaying_factor))

        return self.logit

    def conv(self, x, w, b, conv_stride=(1, 1, 1, 1), conv_padding='SAME', LRN=False, pooling=False, activation=tf.nn.relu):
        ###### 1st conv_gpu1
        res = tf.nn.conv2d(x, w, strides=conv_stride, padding=conv_padding)
        res = tf.nn.bias_add(res, b)

        if LRN :
            res = tf.nn.local_response_normalization(res, depth_radius=self.LRN_depth, bias=self.LRN_bias,
                                                     alpha=self.LRN_alpha, beta=self.LRN_beta)
        if pooling:
            res = tf.nn.max_pool(res, ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='VALID')
        if activation is not None:
            res = activation(res)

        return res
