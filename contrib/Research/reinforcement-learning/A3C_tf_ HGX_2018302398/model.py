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
"""Definenation of the A3C model and the function to choose actions
"""

import tensorflow as tf
import numpy as np
from Params import *
import AcNet


class A3CNet(object):
    def __init__(self, env, scope, sess, OPT_A, OPT_C, ENTROPY_BETA, globalAC=None, lstm=False, input_shape = [None, 80, 80, 1]):
        """Initialize the A3C model
        Args:
            env: the game environment
            scope: scope name
            sess: the running session
            OPT_A: the optimizer of actor
            OPT_C: the optimizer of critic
            ENTROPY_BETA: used to calculate the entropy loss
            globalAC: the global model
            lstm: whether to use LSTM
            input_shape: the shape of input image
        """
        
        self.sess = sess
        
        #The num of actions
        N_A = env.action_space.n
        
        #Define the input node and core net
        self.input = tf.placeholder(tf.float32, input_shape, 's')
        self.acnet = AcNet.AC_Conv(scope, self.input, N_A, self.sess)
        
        #The parameters
        self.a_params, self.c_params = self.acnet.a_params, self.acnet.c_params
        
        #Training structure
        if scope is not GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                #Input while Training
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                td = tf.subtract(self.v_target, self.acnet.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.acnet.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.acnet.a_prob * tf.log(self.acnet.a_prob + 1e-5), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                
                #get the gradients of actor and critic
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            
            #the method to update global model and get parameters from global model
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.acnet.a_params, globalAC.acnet.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.acnet.c_params, globalAC.acnet.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.acnet.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.acnet.c_params))
    
    def update_global(self, feed_dict):
        """using local gradient to update global model"""
        
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):
        """pull parameters from global model to local"""
        
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])
        
    def choose_action(self, s):
        """choose action by local model, random selection based on probability of actions"""
        
        prob = self.acnet.prob_output(s)
        action = np.random.choice(range(prob.shape[1]),p=prob.ravel())  # select action w.r.t the actions prob
        return action
        