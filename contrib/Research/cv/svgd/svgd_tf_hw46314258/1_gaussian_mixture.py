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
# from optimizer import SVGD, Ensemble, AdagradOptimizer
import optimizer
from utils import Time, tf_log_normal
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import ast


# hyper-parameters
num_particles = 100  # number of ensembles (SVGD particles)
num_iterations = 2000  # number of training iterations
learning_rate = 0.01
seed = 0
algorithm = 'svgd'  # either 'svgd' or 'ensemble'

# random seeds
np.random.seed(seed)
tf.set_random_seed(seed)


def network(scope):
    """network"""
    with tf.variable_scope(scope):
        x = tf.Variable(initial_xs[ast.literal_eval(scope[1:])])
        log_prob0, log_prob1 = tf_log_normal(x, -2., 1.), tf_log_normal(x, 2., 1.)
        # log of target distribution p(x)
        log_p = tf.reduce_logsumexp(tf.stack([log_prob0, log_prob1, log_prob1]), axis=0) - tf.log(3.)
        variables_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        gradients = tf.gradients(log_p, variables_)
    return gradients, variables_


def make_gradient_optimizer():
    """make gradient optimizer"""
    return optimizer.AdagradOptimizer(learning_rate=learning_rate)


with Time("graph construction"):
    initial_xs = np.array(np.random.normal(-10, 1, (100,)), dtype=np.float32)

    grads_list, vars_list = [], []
    for i in range(num_particles):
        grads, variables = network('p{}'.format(i))
        grads_list.append(grads)
        vars_list.append(variables)

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

    get_particles_op = tf.trainable_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    with Time("Get initial particles"):
        initial_xs = sess.run(get_particles_op)
    with Time("training"):
        for _ in range(num_iterations):
            sess.run(optimizer.update_op)
    with Time("Get last particles"):
        final_xs = sess.run(get_particles_op)

    # plot
    def plot():
        """plot"""
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        x_grid = np.linspace(-15, 15, 200)

        initial_density = gaussian_kde(initial_xs)
        ax.plot(x_grid, initial_density(x_grid), color='green', label='0th iteration')
        ax.scatter(initial_xs, np.zeros_like(initial_xs), color='green')

        final_density = gaussian_kde(final_xs)
        ax.plot(x_grid, final_density(x_grid), color='red', label='{}th iteration'.format(num_iterations))
        ax.scatter(final_xs, np.zeros_like(final_xs), color='red')

        def log_normal(x, m, s):
            """log normal"""
            return - (x - m) ** 2 / 2. / s ** 2 - np.log(s) - 0.5 * np.log(2. * np.pi)
        target_density = np.exp(log_normal(x_grid, -2., 1.)) / 3 + np.exp(log_normal(x_grid, 2., 1.)) * 2 / 3
        ax.plot(x_grid, target_density, 'r--', label='target density')

        ax.set_xlim([-15, 15])
        ax.set_ylim([0, 0.4])
        ax.legend()
        # plt.show()

    plot()
    print('SVGD')