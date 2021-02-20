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
"""Start the training of the A3C model and save it
"""

import tensorflow as tf
import numpy as np
import threading
import gym
import multiprocessing
from workerm import Worker
from model import A3CNet_LSTM, A3CNet_CONV, A3CNet_DENSE
#from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import matplotlib.pyplot as plt
from tensorflow.python.tools.freeze_graph import freeze_graph
from Params import *
#import moxing as mox
from envs import create_atari_env
import os
import argparse
import time
#from npu_bridge.estimator.npu.npu_config import DumpConfig

tf = tf.compat.v1

def parse_args():
    """get parameters from commands"""
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_url', default='./model/',
                        help="""output directory""")
    parser.add_argument('--log_url', default='./log',
                        help="""log directory""")
    parser.add_argument('--mox_url', default='',
                        help="""obs directory""")
    parser.add_argument('--env_name', default='PongDeterministic-v4',
                        help="""name of training env""")
    parser.add_argument('--threads_num', type=int, default=4,
                        help="""number of threads""")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument('--UPDATE_GLOBAL_ITER', type=int, default=256,
                        help="""value of UPDATE_GLOBAL_ITER""")
    parser.add_argument('--MAX_GLOBAL_EP', type=int, default=10000,
                        help="""value of MAX_GLOBAL_EP""")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="""learning rate of OPT""")
    parser.add_argument('--GAMMA', type=float, default=0.9,
                        help="""value of GAMMA""")
    parser.add_argument('--ENTROPY_BETA', type=float, default=0.01,
                        help="""value of ENTROPY_BETA""")
    parser.add_argument('--LSTM', action='store_true',
                        help="""Wether to use lstm""")
    parser.add_argument('--model_name', default='model',
                        help="""name of saved model""")
    
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    
    return args

def main():

    #get parameters from commands
    args = parse_args()
    output_url = args.output_url
    log_url = args.log_url
    mox_url = args.mox_url
    game_Name = args.env_name
    threads_num = args.threads_num
    lr = args.lr
    seed = args.seed
    model_name = args.model_name + ".ckpt"
    
    UPDATE_GLOBAL_ITER = args.UPDATE_GLOBAL_ITER
    MAX_GLOBAL_EP = args.MAX_GLOBAL_EP
    GAMMA = args.GAMMA
    ENTROPY_BETA = args.ENTROPY_BETA
    LSTM = args.LSTM

    GLOBAL_RUNNING_R = []
    LOG = []
    
    #make output directory if it is not exist
    if not os.path.isdir(output_url):
        os.makedirs(output_url)
    if not os.path.isdir(log_url):
        os.makedirs(log_url)
    
    #define the config of session to train on npu
    #config = tf.ConfigProto()
    #custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    #custom_op.name =  "NpuOptimizer"
    #custom_op.parameter_map["use_off_line"].b = False #在昇腾AI处理器执行训练
    #config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关 
    
    
    with tf.Session() as sess:
        T0 = time.time()
        
        #define the global model

        env = create_atari_env(game_Name)
        shape = env.observation_space.shape
        
        #OPT = tf.train.RMSPropOptimizer(lr, name='RMSPropC')
        OPT = tf.train.AdamOptimizer(lr, name='Adamopt')
        
        if LSTM:
            global_Net = A3CNet_LSTM(env, GLOBAL_NET_SCOPE, sess, OPT, ENTROPY_BETA, input_shape=[None, shape[1], shape[2], shape[0]])
        else:
            global_Net = A3CNet_CONV(env, GLOBAL_NET_SCOPE, sess, OPT, ENTROPY_BETA, input_shape=[None, shape[1], shape[2], shape[0]])
            
        #define workers
        workers = []
        max_threading_nums = threads_num
        coord = tf.train.Coordinator()
        for rank in range(max_threading_nums):
            workers.append(Worker(sess, game_Name, f'W_{rank}', seed, rank, global_Net, OPT, coord, UPDATE_GLOBAL_ITER, MAX_GLOBAL_EP, GAMMA, ENTROPY_BETA, lstm=LSTM))
        sess.run(tf.global_variables_initializer())
        
        #start training in different threads
        thread_pool = []
        for worker in workers:
            job = lambda: worker.work(GLOBAL_RUNNING_R, LOG, log_url)
            t = threading.Thread(target=job)
            t.start()
            time.sleep(0.1)
            thread_pool.append(t)
        coord.join(thread_pool)
        
        #save the model and log to target directory
        f = open(os.path.join(log_url, "log.txt"), 'a')
        for item in LOG:
            (name, G_EP, ep_r, Score, frames_local, frames, t) = item
            t = t-T0
            f.write("Worker Name: %s, Episode: %d, Local Frames: %d, Total Frames: %d, Local Score: %d, Total Score: %.2f, Time: %.2f \n" % (name, G_EP, frames_local, frames, ep_r, Score, t))

        f.close()
        
        var_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, os.path.join(output_url, model_name))
        
    #mox.file.copy_parallel("./", mox_url)

if __name__=='__main__':
    main()

