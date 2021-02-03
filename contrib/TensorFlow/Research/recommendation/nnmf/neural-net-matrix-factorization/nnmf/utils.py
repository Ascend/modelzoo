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

#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Various utils."""
# Standard modules
import math
# Third party modules
import tensorflow as tf

def KL(mean, log_var, prior_var):
    """Computes KL divergence for a group of univariate normals (ie. every dimension of a latent)."""
    return tf.reduce_sum(tf.log(math.sqrt(prior_var) / tf.sqrt(tf.exp(log_var))) + (
    (tf.exp(log_var) + tf.square(mean)) / (2.0 * prior_var)), reduction_indices=[0, 1])

def _weight_init_range(n_in, n_out):
    """Calculates range for picking initial weight values from a uniform distribution."""
    range = 4.0 * math.sqrt(6.0) / math.sqrt(n_in + n_out)
    return {
        'minval': -range,
        'maxval': range,
    }

def build_mlp(f_input_layer, hidden_units_per_layer):
    """Builds a feed-forward NN (MLP) with 3 hidden layers."""
    # Note: tf.contrib.layers could likely be used instead, but total control allows for easier debugging in this case
    # TODO make number of hidden layers a parameter, if needed
    num_f_inputs = f_input_layer.get_shape().as_list()[1]

    # MLP weights picked uniformly from +/- 4*sqrt(6)/sqrt(n_in + n_out)
    mlp_weights = {
        'h1': tf.Variable(tf.random_uniform([num_f_inputs, hidden_units_per_layer],
                                            **_weight_init_range(num_f_inputs, hidden_units_per_layer))),
        'b1': tf.Variable(tf.zeros([hidden_units_per_layer])),
        'h2': tf.Variable(tf.random_uniform([hidden_units_per_layer, hidden_units_per_layer],
                                            **_weight_init_range(hidden_units_per_layer, hidden_units_per_layer))),
        'b2': tf.Variable(tf.zeros([hidden_units_per_layer])),
        'h3': tf.Variable(tf.random_uniform([hidden_units_per_layer, hidden_units_per_layer],
                                            **_weight_init_range(hidden_units_per_layer, hidden_units_per_layer))),
        'b3': tf.Variable(tf.zeros([hidden_units_per_layer])),
        'out': tf.Variable(tf.random_uniform([hidden_units_per_layer, 1],
                                            **_weight_init_range(hidden_units_per_layer, 1))),
        'b_out': tf.Variable(tf.zeros([1])),
    }
    # MLP layers
    mlp_layer_1 = tf.nn.sigmoid(tf.matmul(f_input_layer, mlp_weights['h1']) + mlp_weights['b1'])
    mlp_layer_2 = tf.nn.sigmoid(tf.matmul(mlp_layer_1, mlp_weights['h2']) + mlp_weights['b2'])
    mlp_layer_3 = tf.nn.sigmoid(tf.matmul(mlp_layer_2, mlp_weights['h3']) + mlp_weights['b3'])
    out = tf.matmul(mlp_layer_3, mlp_weights['out']) + mlp_weights['b_out']

    return out, mlp_weights


def get_kl_weight(curr_iter, on_epoch=100):
    """Outputs sigmoid scheduled KL weight term (to be fully on at 'on_epoch')"""
    return 1.0 / (1 + math.exp(-(25.0 / on_epoch) * (curr_iter - (on_epoch / 2.0))))


def chunk_df(df, size):
    """Splits a Pandas dataframe into chunks of size `size`.

    See here: https://stackoverflow.com/a/25701576/1424734
    """
    #return (df[pos:pos + size] for pos in xrange(0, len(df), size))
    return (df[pos:pos + size] for pos in range(0, len(df), size))
