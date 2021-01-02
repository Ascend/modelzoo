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
from model import A3CNet
from Params import *
import time
import gym
import cv2
import sys
import os

def downSample(s):
    """Preprocessing to the image of game state"""
    s = s[35:195]
    s = s[::2,::2, 0]
    s[s==144] = 0
    s[s==109] = 0
    s[s!=0] = 1
    return s[:,:,np.newaxis]

class Worker(object):
    """Define the Worker class, it canbe used in model training"""
    
    def __init__(self, sess, GameName, name, globalAC, OPT_A, OPT_C, coord, UPDATE_GLOBAL_ITER, MAX_GLOBAL_EP, GAMMA, ENTROPY_BETA, lstm=False):
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
        self.env = gym.make(GameName).unwrapped
        self.name = name
        self.AC = A3CNet(self.env, name, sess, OPT_A, OPT_C, ENTROPY_BETA, globalAC, lstm)
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
        
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        
        while GLOBAL_EP < self.MAX_GLOBAL_EP and not self.coord.should_stop(): #
            
            s = self.env.reset()
            ep_r = 0
            s = downSample(s)
            while True:
                FRAME_TOTAL += 1
                frames += 1
                s = s[np.newaxis, :]
                
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                ep_r += r
                
                s_ = downSample(s_)
                
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                
                #update the global model
                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    elif len(self.env.observation_space.shape) == 3:
                        v_s_ = self.sess.run(self.AC.acnet.v, {self.AC.acnet.input: s_[np.newaxis, :]})[0, 0]
                    else:
                        v_s_ = self.sess.run(self.AC.acnet.v, {self.AC.acnet.input: s_[np.newaxis, np.newaxis, :, np.newaxis]})[0, 0]
                        
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
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
                    
                    log = (self.name, GLOBAL_EP, GLOBAL_RUNNING_R[-1], frames, time.time()-self.T0)
                    LOG.append(log)
                    print(log)
                    GLOBAL_EP += 1
                    break

                    #infer the performance of an model
    def test_score(self, env, num, model):
        """Use local model to perform on the games, and get the scores."""
        
        score_tot = 0
        for i in range(num):
            s = env.reset()
            s = downSample(s)
            ep_r = 0
            while(True):
                s = s[np.newaxis, :]
                a = model.choose_action(s)
                s_, r, done, info = env.step(a)
                ep_r += r
                s_ = downSample(s_)
                s = s_
                
                if done:
                    score_tot += ep_r
                    break
        return score_tot/num