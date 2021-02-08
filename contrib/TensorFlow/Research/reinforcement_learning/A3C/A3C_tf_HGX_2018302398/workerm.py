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
"""
Definition of Worker, which is used as the basic structure to train a model.
"""

import tensorflow as tf
import numpy as np
from model import A3CNet_LSTM, A3CNet_DENSE, A3CNet_CONV
from Params import *
from envs import create_atari_env
import time
import gym
import cv2
import sys
import os


class Worker(object):
    """Define the Worker class, it canbe used in model training"""
    
    def __init__(self, sess, GameName, name, seed, rank, globalAC, OPT, coord, UPDATE_GLOBAL_ITER, MAX_GLOBAL_EP, GAMMA, ENTROPY_BETA, lstm=True):
        """Initialize the object of the class Worker.
        Args:
            sess: the running session
            GameName: name of game
            name: name of the worker/scope
            globalAC: the global net
            OPT_A: the optimizer of actor
            OPT_C: the optimizer of critic
            coord: training coordinator
            UPDATE_GLOBAL_ITER: the number of updating steps
            MAX_GLOBAL_EP: the maximum of steps
            GAMMA: decay rate on the contribution of past scores
            ENTROPY_BETA: used to calculate the entropy loss
            lstm: whether to use LSTM
        """
        
        self.sess = sess
        self.name = name
        self.lstm = lstm
        self.env = create_atari_env(GameName)
        self.env.seed(seed + rank)
        shape = self.env.observation_space.shape
        if lstm:
            self.AC = A3CNet_LSTM(self.env, name, sess, OPT, ENTROPY_BETA, globalAC, input_shape=[None, shape[1], shape[2], shape[0]])
        else:
           self.AC = A3CNet_CONV(self.env, name, sess, OPT, ENTROPY_BETA, globalAC, input_shape=[None, shape[1], shape[2], shape[0]])
        self.coord = coord
        self.T0 = time.time()
        self.UPDATE_GLOBAL_ITER = UPDATE_GLOBAL_ITER
        self.MAX_GLOBAL_EP = MAX_GLOBAL_EP
        self.GAMMA = GAMMA
        
    def work(self, GLOBAL_RUNNING_R, LOG, log_url):
        """Start the training process, infer with local model and then update the global model
        Args:
            GLOBAL_RUNNING_R: record the past scores
            LOG: record of other parameters in training
            log_url: the directory of log files
        """
        
        global GLOBAL_EP
        global FRAME_TOTAL
        frames = 0
        
        #save the log(structure)
        path = os.path.join(log_url, self.name)
        if not os.path.isdir(path):
            os.makedirs(path)
        
        writer = tf.summary.FileWriter(path, self.sess.graph)
        if self.lstm:
            lstm_state = self.AC.acnet.state_init
        
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        
        while GLOBAL_EP < self.MAX_GLOBAL_EP and not self.coord.should_stop(): #
            
            s = self.env.reset()
            s = np.squeeze(s)[np.newaxis, :, :, np.newaxis]
            ep_r = 0
            frame_local = 0
            while True:
                
                FRAME_TOTAL += 1
                frames += 1
                frame_local += 1
                if self.lstm:
                    a, lstm_state = self.AC.choose_action(s, lstm_state)
                else:
                    a = self.AC.choose_action(s)
                
                s_, r, done, info = self.env.step(a)
                ep_r += r
                
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                
                s_ = np.squeeze(s_)[np.newaxis, :, :, np.newaxis]
                
                #update the global model
                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    elif self.lstm:
                        v_s_ = self.sess.run(self.AC.acnet.v, {
                                            self.AC.acnet.input: s_,
                                            self.AC.acnet.c_in: lstm_state[0],
                                            self.AC.acnet.h_in: lstm_state[1]})[0, 0]
                    else:
                        v_s_ = self.sess.run(self.AC.acnet.v, {
                                            self.AC.acnet.input: s_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    if self.lstm:
                        feed_dict = {
                            self.AC.acnet.input: buffer_s,
                            self.AC.a_his: buffer_a,
                            self.AC.v_target: buffer_v_target,
                            self.AC.acnet.c_in: lstm_state[0],
                            self.AC.acnet.h_in: lstm_state[1],
                        }
                    else:
                        feed_dict = {
                            self.AC.acnet.input: buffer_s,
                            self.AC.a_his: buffer_a,
                            self.AC.v_target: buffer_v_target,
                        }
                    
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    
                s = s_
                total_step += 1
                
                #record the parameters after an epoch
                if done:
                    
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    
                    log = (self.name, GLOBAL_EP, ep_r, GLOBAL_RUNNING_R[-1], frame_local, frames, time.time()-self.T0)
                    LOG.append(log)
                    Str = "Worker Name: %s, Episode: %d, Local Frames: %d, Total Frames: %d, Local Score: %d, Total Score: %.2f, Time: %.2f" % (self.name, GLOBAL_EP, frame_local, frames, ep_r, GLOBAL_RUNNING_R[-1], time.time()-self.T0)
                    print(Str)
                    GLOBAL_EP += 1
                    break

                    #infer the performance of an model