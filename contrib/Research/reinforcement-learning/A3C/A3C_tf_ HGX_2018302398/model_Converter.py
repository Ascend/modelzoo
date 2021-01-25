import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import envs
import argparse

import cv2
from tensorflow.python.tools import freeze_graph 
from tensorflow.python.platform import gfile

from envs import create_atari_env

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
            self._build_net(scope, classNum)
            
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
    
    def output(self, s):
        prob_weights = self.sess.run(self.a_prob, 
                                      feed_dict={self.input: s})
        return prob_weights

class AC_LSTM(object):
    """the structure of core net"""
    
    def __init__(self, scope, input, classNum, sess):
        
        self.sess = sess
        self.input = input
        
        with tf.variable_scope(scope):
            self._build_net(scope, classNum)
            
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
            
    #get the output node of core net
    #the probability of actions
    def output(self, s, lstm_state):
        prob_weights, state_out = self.sess.run([self.a_prob, self.state_out], 
                                      feed_dict={self.input: s, 
                                                 self.c_in: lstm_state[0],
                                                 self.h_in: lstm_state[1]})
        return prob_weights, state_out

def save_conv(SESS, output_path, model_name, ckpt_path):
    tf.train.write_graph(SESS.graph_def, output_path, 'tmp.pb') 
    freeze_graph.freeze_graph(
        input_graph = output_path+'/tmp.pb', 
        input_saver='', 
        input_binary=False, 
        input_checkpoint=ckpt_path, 
        output_node_names='out_a',
        restore_op_name='save/restore_all', 
        filename_tensor_name='save/Const:0', 
        output_graph = output_path+"/"+model_name+".pb",
        clear_devices=False, 
        initializer_nodes=''
    )
    print("convert sucess")

def save_lstm(SESS, output_path, model_name, ckpt_path):
    tf.train.write_graph(SESS.graph_def, output_path, 'tmp.pb') 
    freeze_graph.freeze_graph(
        input_graph = output_path+'/tmp.pb', 
        input_saver='', 
        input_binary=False, 
        input_checkpoint=ckpt_path, 
        output_node_names='out_a, W_0/actor/rnn/while/Exit_3, W_0/actor/rnn/while/Exit_4',
        restore_op_name='save/restore_all', 
        filename_tensor_name='save/Const:0', 
        output_graph = output_path+"/"+model_name+".pb", 
        clear_devices=False, 
        initializer_nodes=''
    )
    print("sucess")
        
def parse_args():
    """get parameters from commands"""
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--output_path', default='./pb_model/',
                        help="""output directory""")
    parser.add_argument('--input_path', default='./pb_model/',
                        help="""input directory""")
    parser.add_argument('--model_name', default='model',
                        help="""name of the model""")
    parser.add_argument('--type', default='conv',
                        help="""type of the model""")
    parser.add_argument('--env_name', default='PongDeterministic-v4',
                        help="""name of the environment""")
    
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    
    return args

def main():
    args = parse_args()
    
    output_path = args.output_path
    input_path = args.input_path
    model_name = args.model_name
    type = args.type
    env_name = args.env_name
    ckpt_path = input_path+"/"+model_name+".ckpt"
    
    print(ckpt_path)
    env = create_atari_env(env_name)
    shape = env.observation_space.shape
    N_A = env.action_space.n
    
    input = tf.placeholder(tf.float32, [1, 42, 42, 1], 's')
    SESS = tf.Session()
    
    if type == "conv":
        model = AC_CONV("W_0", input, N_A, SESS)
    else:
        model = AC_LSTM("W_0", input, N_A, SESS)
    
    flow = model.a_prob
    flow = tf.argmax(flow, axis=1, output_type=tf.int32, name='out_a')
    
    var = tf.global_variables()
    saver = tf.train.Saver(var)
    SESS.run(tf.global_variables_initializer())

    if type == "conv":
        save_conv(SESS, output_path, model_name, ckpt_path)
    else:
        save_lstm(SESS, output_path, model_name, ckpt_path)
        
if __name__=="__main__":
    main()