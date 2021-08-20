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
# Copyright (c) 2018 by huyz. All Rights Reserved.

# Reference: [CVPR2018] Non-local Neural Networks

from npu_bridge.npu_init import *
import argparse
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# 鎺ュ彈鍙傛暟,鍙傛暟鍜屾暟鍊间箣闂翠互绗﹀彿闅斿紑
def parse_args():
    desc = 'MAIN'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='', help='dataset_name')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    return parser.parse_args()

args = parse_args()

data_path = args.data_path
mnist = input_data.read_data_sets(data_path, one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

X_img = tf.pad(X_img, [[0, 0], [98, 98], [98, 98], [0, 0]])

learning_rate = 0.0001
#batch_size = 128
#batch_size = 16
batch_size = args.batch_size
#num_epoches = 2
num_epoches = args.epoch

def global_avg_pool(x):
    axis = [1, 2]
    return tf.reduce_mean(x, axis, keep_dims=True)


def fc(x, output_units):
    x = tf.reshape(x, shape=[-1, x.get_shape()[-1]])
    x = tf.layers.dense(inputs=x,
                        units=output_units)
    return x


def relu(x):
    return tf.nn.relu(x)


def batch_norm(name, x, is_training=True):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(inputs=x,
                                    axis=3,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    training=is_training,
                                    fused=True)


def padding(x, kernel_size):
    # Padding based on kernel_size
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return x


def pad_conv(name, x, kernel_size, output_channels, strides, bias=False):
    if strides > 1:
        x = padding(x, kernel_size)
    with tf.variable_scope(name):
        x = tf.layers.conv2d(inputs=x,
                            filters=output_channels,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=("same" if strides == 1 else "valid"),
                            use_bias=bias,
                            kernel_initializer=tf.variance_scaling_initializer())
        return x


def NonLocalBlock(input_x, output_channels, sub_sample=False, is_bn=True, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope):
        with tf.variable_scope("g"):
            g = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="g_conv")
            if sub_sample:
                g = tf.layers.max_pooling2d(inputs=g, pool_size=2, strides=2, padding="valid", name="g_max_pool")
                print(g.shape)

        with tf.variable_scope("phi"):
            phi = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="phi_conv")
            if sub_sample:
                phi = tf.layers.max_pooling2d(inputs=phi, pool_size=2, strides=2, padding="valid", name="phi_max_pool")
                #print(phi.shape)

        with tf.variable_scope("theta"):
            theta = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="theta_conv")
            print(theta.shape)

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(g, [-1, height * width, output_channels])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(phi, [-1, output_channels, height * width])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(theta, [-1, height * width, output_channels])
        print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)      
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [-1, height, width, output_channels])

        with tf.variable_scope("w"):
            w_y = tf.layers.conv2d(inputs=y, filters=in_channels, kernel_size=1, strides=1, padding="same", name="w_conv")
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_x + w_y

        return z


def residual(x, output_channels, strides, type, is_training):
    # type: short cut type, "conv" or "identity"
    short_cut = x

    # short_cut
    if type == "conv":
        short_cut = batch_norm("conv1_b1_bn", short_cut, is_training)
        short_cut = relu(short_cut)
        short_cut = pad_conv("conv1_b1", short_cut, 1, output_channels, strides)

    # bottleneck residual block
    x = batch_norm("conv1_b2_bn", x, is_training)
    x = relu(x)
    x = pad_conv("conv1_b2", x, 1, output_channels/4, 1)
    x = batch_norm("conv2_b2_bn", x, is_training)
    x = relu(x)
    x = pad_conv("conv2_b2", x, 3, output_channels/4, strides)
    x = batch_norm("conv3_b2_bn", x, is_training)
    x = relu(x)
    x = pad_conv("conv3_b2", x, 1, output_channels, 1)

    return x + short_cut


def Build_ResNet(x, resnet_size, is_training = True):
    output_channels = [256, 512, 1024, 2048]
    strides = [2, 2, 2, 2]

    if resnet_size == 50:
        stages = [3, 4, 6, 3]
    elif resnet_size == 101:
        stages = [3, 4, 23, 3]
    elif resnet_size == 152:
        stages = [3, 8, 36, 3]
    else:
        raise ValueError("resnet_size %d not implement"%resnet_size)

    # init net
    with tf.variable_scope("init"):
        x = pad_conv("init_conv", x, 7, 64, 2)

    # 4 stages
    for i in range(len(stages)):
        with tf.variable_scope("stage_%d_block_%d"%(i, 0)):
            if stages[i] == 4 or stages[i] == 6:
                x = residual(x, output_channels[i], strides[i], "conv", is_training)
                x = NonLocalBlock(x, output_channels[i], scope="Non-local_%d"%i)
            else:
                x = residual(x, output_channels[i], strides[i], "conv", is_training)

            for j in range(1, stages[i]):
                with tf.variable_scope("stage_%d_block_%d"%(i, j)):
                    if stages[i] == 4 or stages[i] == 6:
                        x = residual(x, output_channels[i], 1, "identity", is_training)
                        x = NonLocalBlock(x, output_channels[i], scope="Non_local_%d"%j)
                    else:
                        x = residual(x, output_channels[i], 1, "identity", is_training)

    with tf.variable_scope("global_pool"):
        x = batch_norm("bn", x, is_training)
        x = relu(x)
        x = global_avg_pool(x)

    with tf.variable_scope("logit"):
        logit = fc(x, 10)

    return logit


logit = Build_ResNet(X_img, resnet_size=50)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
cost_summ = tf.summary.scalar("cost", cost)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

is_correction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

############################## npu modify #########################
init = tf.global_variables_initializer()

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #off remap
config = npu_config_proto(config_proto=config)

# with tf.Session() as sess:
with tf.Session(config=config) as sess:
    sess.run(init)
############################## npu modify #########################

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./log")
    writer.add_graph(sess.graph)

    print("Learning start...")

    for epoch in range(num_epoches):
        avg_acc = 0
        avg_cost = 0

        num_batches = int(mnist.train.num_examples / batch_size)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}

        for i in range(num_batches):
            start_time = time.time()
            summary, _, c, a = sess.run([merged_summary, optimizer, cost, accuracy], feed_dict=feed_dict)
            writer.add_summary(summary, global_step=i)
            acc = a
            cost_ = c
            end_time = time.time()
            steps_per_s = 1/(end_time - start_time)

            #print("Epoch: {}\tLoss:{:.9f}\tAccuarcy: {:.2%}\tglobal_step/sec:{}".format(epoch+1, avg_cost, avg_acc, steps_per_s))
            print("Epoch: {}\tLoss:{:.9f}\tAccuarcy: {:.4}\tglobal_step/sec:{}".format(epoch+1, cost_, acc, steps_per_s))
print("Learning finished!")


'''
if __name__ == "__main__":
    x = tf.Variable(tf.random_normal([1, 224, 224, 3]))
    x = Build_ResNet(x, 50)
    print(x.get_shape().as_list())
'''

