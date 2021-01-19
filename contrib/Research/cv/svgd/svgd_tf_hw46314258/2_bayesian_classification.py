"""This is an experiment"""
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
# Copyright 2020 Huawei Technologies Co., Ltd
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


import tensorflow as tf
import numpy as np
from utils import Time
# from optimizer import SVGD, Ensemble
import optimizer
import matplotlib.pyplot as plt

# hyper-parameters
num0 = 200  # number of samples in class0 (<400)
num_particles = 20  # number of ensembles (SVGD particles)
num_iterations = 500  # number of training iterations
seed = 0
algorithm = 'svgd'  # 'svgd' or 'ensemble'

# random seeds
np.random.seed(seed)
tf.set_random_seed(seed)

# data generation
num1 = 400 - num0
mean0 = np.array([-1, -1])
std0 = np.array([0.25, 0.25])
mean1 = np.array([1, 1])
std1 = np.array([1.5, 1.5])
x0 = np.tile(mean0, (num0, 1)) + std0 * np.random.randn(num0, 2)
x1 = np.tile(mean1, (num1, 1)) + std1 * np.random.randn(num1, 2)
y0 = np.zeros((x0.shape[0], 1))
y1 = np.ones((x1.shape[0], 1))

x = np.concatenate([x0, x1], axis=0)
y = np.concatenate([y0, y1], axis=0)
D = np.hstack([x, y])
np.random.shuffle(D)
x = np.array(D[:, 0:2], dtype=np.float32)
y = np.array(D[:, 2:], dtype=np.float32)
x_train = x[:300]
y_train = y[:300]
x_test = x[300:]
y_test = y[300:]


def network(inputs, labels, scope):
    """network"""
    net = inputs
    # See /derivations/bayesian_classification.pdf for mathematical details.
    with tf.variable_scope(scope):
        for _ in range(2):
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh)
        logits = tf.layers.dense(net, 1)
        log_likelihood = - tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        variables_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        prob_1_x_w_ = tf.nn.sigmoid(logits)
        gradients = tf.gradients(log_likelihood, variables_)
    return gradients, variables_, prob_1_x_w_


def make_gradient_optimizer():
    """make gradient optimizer"""
    return tf.train.AdamOptimizer(learning_rate=0.001)


with Time("graph construction"):
    x_, y_ = tf.placeholder(tf.float32, [None, 2]), tf.placeholder(tf.float32, [None, 1])

    grads_list, vars_list, prob_1_x_w_list = [], [], []
    for i in range(num_particles):
        grads, variables, prob_1_x_w = network(x_, y_, 'p{}'.format(i))
        grads_list.append(grads)
        vars_list.append(variables)
        prob_1_x_w_list.append(prob_1_x_w)

    if algorithm == 'svgd':
        optimizer = optimizer.SVGD(grads_list=grads_list,
                         vars_list=vars_list,
                         make_gradient_optimizer=make_gradient_optimizer)
    elif algorithm == 'ensemble':
        optimizer = optimizer.Ensemble(grads_list=grads_list,
                             vars_list=vars_list,
                             make_gradient_optimizer=make_gradient_optimizer)
    else:
        raise NotImplementedError

    prob_1_x = tf.reduce_mean(tf.stack(prob_1_x_w_list), axis=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    with Time("training"):
        for _ in range(num_iterations):
            sess.run(optimizer.update_op, feed_dict={x_: x_train, y_: y_train})

    with Time("test"):
        p1 = sess.run(prob_1_x, feed_dict={x_: x_test})
        classification = np.array(p1) > 0.5
        error_rate = np.sum(classification != y_test) / y_test.shape[0] * 100
        print('Error rate: {}%'.format(error_rate))

    # plot
    def plot():
        """plot"""
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        x0_grid, x1_grid = np.linspace(-7, 7, 50), np.linspace(-7, 7, 50)
        x0_grid, x1_grid = np.meshgrid(x0_grid, x1_grid)

        x_grid = np.hstack([x0_grid.reshape(-1, 1), x1_grid.reshape(-1, 1)])
        p1_grid = sess.run(prob_1_x, feed_dict={x_: x_grid}).reshape(x0_grid.shape)
        contour = ax.contour(x0_grid, x1_grid, p1_grid, 50, cmap=plt.cm.coolwarm, zorder=0)

        x_0, x_1 = x_train[np.where(y_train[:, 0] == 0)], x_train[np.where(y_train[:, 0] == 1)]
        ax.scatter(x_0[:, 0], x_0[:, 1], s=1, c='blue', zorder=1)
        ax.scatter(x_1[:, 0], x_1[:, 1], s=1, c='red', zorder=2)

        ax.set_title('$p(1|(x_0, x_1))$ with {} ({} particles)'.format(algorithm, num_particles))
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal', 'box')
        ax.grid(b=True)
        fig.colorbar(contour)
        plt.show()

    plot()
