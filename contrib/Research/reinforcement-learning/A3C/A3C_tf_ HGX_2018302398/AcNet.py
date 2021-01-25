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

class AC_DENSE(object):
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
            self.params = self._build_net(scope, classNum)
            
    def _build_net(self, scope, classNum):
        """The detailed structure of core net"""
        w_init = tf.random_normal_initializer(0., .1)

        self.flat = tf.layers.flatten(self.input, name='flat')
        
        #define the actor
        with tf.variable_scope('actor'):
            self.a_dense = tf.layers.dense(self.flat, 100, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.a_prob = tf.layers.dense(self.a_dense, classNum, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            
        #define the critic
        with tf.variable_scope('critic'):
            self.c_dense = tf.layers.dense(self.flat, 64, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.v = tf.layers.dense(self.flat, 1, kernel_initializer=w_init, name='v')
        
        #the parameters in core net.
        #Using in the procession of network updating and gradient calculation
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return params

    #get the output node of core net
    #the probability of actions
    def output(self, s):
        prob_weights, value = self.sess.run([self.a_prob, self.v], 
                                      feed_dict={self.input: s})
        return prob_weights

class AC_CONV(object):
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
            self.params = self._build_net(scope, classNum)
            
    def _build_net(self, scope, classNum):
        """The detailed structure of core net"""
        w_init = tf.random_normal_initializer(0., .1)
        self.conv1 = tf.layers.conv2d(self.input, filters=32, kernel_size=3, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv1')
        self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv2')
        self.conv3 = tf.layers.conv2d(self.conv2, filters=32, kernel_size=3, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv3')
        
        self.flat = tf.layers.flatten(self.conv3, name='flat')
        
        #define the actor
        with tf.variable_scope('actor'):
            self.a_dense = tf.layers.dense(self.flat, 256, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.a_prob = tf.layers.dense(self.a_dense, classNum, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            
        #define the critic
        with tf.variable_scope('critic'):
            self.c_dense = tf.layers.dense(self.flat, 128, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            self.v = tf.layers.dense(self.c_dense, 1, kernel_initializer=w_init, name='v')
        
        #the parameters in core net.
        #Using in the procession of network updating and gradient calculation
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return params

    #get the output node of core net
    #the probability of actions
    def output(self, s):
        prob_weights, value = self.sess.run([self.a_prob, self.v], 
                                      feed_dict={self.input: s})
        return prob_weights

class AC_LSTM(object):
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
            #self.a_params, self.c_params = self._build_net(scope, classNum)
            self.params = self._build_net(scope, classNum)
            
    def _build_net(self, scope, classNum):
        """The detailed structure of core net"""

        #define the actor
        with tf.variable_scope('actor'):
            w_init = tf.random_normal_initializer(0., .1)
            self.conv1 = tf.layers.conv2d(self.input, filters=32, kernel_size=3, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv1')
            self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv2')
            self.conv3 = tf.layers.conv2d(self.conv2, filters=32, kernel_size=3, strides=2, padding='VALID', activation=tf.nn.relu, name = 'conv3')
            
            flat = tf.layers.flatten(self.conv3, name='flat0')
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True, name='lstm_cell')
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            
            self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name='c_in')
            self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name='h_in')
                
            rnn_in = tf.expand_dims(flat, [0], name='flat_exp')
            state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in,
                    initial_state=state_in,
                    time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.rnn_out = tf.reshape(lstm_outputs, [-1, 256], name='reshape')
            
            self.flat = tf.layers.flatten(self.rnn_out, name='flat')
            
            self.a_prob = tf.layers.dense(self.flat, classNum, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        
        #define the critic
        with tf.variable_scope('critic'):
            self.v = tf.layers.dense(self.flat, 1, kernel_initializer=w_init, name='v')  # state value
        
        #the parameters in core net.
        #Using in the procession of network updating and gradient calculation
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return params

    #get the output node of core net
    #the probability of actions
    def output(self, s, lstm_state):
        prob_weights, value, state_out = self.sess.run([self.a_prob, self.v, self.state_out], 
                                      feed_dict={self.input: s, 
                                                 self.c_in: lstm_state[0],
                                                 self.h_in: lstm_state[1]})
        return prob_weights, state_out