# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python2, python3
""" The structure of network used in the model
"""
import tensorflow as tf
import numpy as np
from Params import *

class AC_Conv(object):
    """the structure of core net"""
    
    def __init__(self, scope, input, classNum, sess):
        """Initialize the core net
        Args:
            scope: scope name
            input: input node
            classNum: the number of actions
            sess: the activated session
        """
        
        self.sess = sess
        self.input = input
        
        with tf.variable_scope(scope):
            self.a_params, self.c_params = self._build_net(scope, classNum)
            
    def _build_net(self, scope, classNum):
        """The detailed structure of core net"""
        
        w_init = tf.random_normal_initializer(0., .1)
        self.conv1 = tf.layers.conv2d(self.input, filters=16, kernel_size=8, strides=4, padding='VALID', activation=tf.nn.relu, name = 'conv1')
        self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=4, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv2')
        self.conv3 = tf.layers.conv2d(self.conv2, filters=32, kernel_size=3, strides=1, padding='VALID', activation=tf.nn.relu, name = 'conv3')
        
        self.flat = tf.layers.flatten(self.conv1, name='flat')
        
        #define the actor
        with tf.variable_scope('actor'):
            self.a_dense = tf.layers.dense(self.flat, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.a_prob = tf.layers.dense(self.a_dense, classNum, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        
        #define the critic
        with tf.variable_scope('critic'):
            self.c_dense = tf.layers.dense(self.flat, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            self.v = tf.layers.dense(self.c_dense, 1, kernel_initializer=w_init, name='v')  # state value
        
        #the parameters in core net.
        #Using in the procession of network updating and gradient calculation
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_params, c_params

    #get the output node of core net
    #the probability of actions
    def prob_output(self, s):
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.input: s})
        return prob_weights