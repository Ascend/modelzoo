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

import os
import gym
import numpy as np
from replay_memory import ReplayBuffer, PrioritizedReplayBuffer
import math
import random
import pickle
import time
from collections import deque
from wrappers import wrap, wrap_cover, SubprocVecEnv
import tensorflow as tf
from tensorflow.contrib import slim
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.compat import compat
# Parameters
import argparse
parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('--games', type=str, default="Pong", help='name of the games. for example: Breakout')
parser.add_argument('--data_url', default=None, type=str, help='input_dir')
parser.add_argument('--train_url', default='./model_save', type=str, help='output_dir')
parser.add_argument('--load', default=0, type=int)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--loss_scale', default=1024, type=int, help='loss scale')
parser.add_argument('--target_replace_iter', default=100, type=int, help='target policy sync interval')
parser.add_argument('--memory_capacity', default=int(1e+5), type=int, help='experience replay memory size')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--envs_num', default=32, type=int, help='envs num')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='envs num')
parser.add_argument('--quant_num', default=64, type=int, help='quant num')
parser.add_argument('--step_num', default=int(4e+7), type=int, help='step_num')
parser.add_argument('--save_freq', default=1000, type=int, help='save_freq')
args = parser.parse_args()
args.games = "".join(args.games)

'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = args.target_replace_iter
# simulator steps for start learning
LEARN_START = int(1e+3)
# (prioritized) experience replay memory size
MEMORY_CAPACITY = args.memory_capacity
# simulator steps for learning interval
LEARN_FREQ = 4
# quantile numbers for IQN
N_QUANT = args.quant_num
# quantiles
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]

'''Environment Settings'''
# number of environments for C51
N_ENVS = args.envs_num
# Total simulation step
STEP_NUM = args.step_num
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False
# openai gym env name
# ENV_NAME = 'BreakoutNoFrameskip-v4'
ENV_NAME = args.games + 'NoFrameskip-v4'
env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(N_ENVS)])
N_ACTIONS = env.action_space.n

'''Training settings'''
# check GPU usage
# mini-batch size
BATCH_SIZE = args.batch_size
# learning rage
LR = args.learning_rate
# epsilon-greedy
EPSILON = 1.0

'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = args.load
# save frequency
SAVE_FREQ = args.save_freq
# paths for predction net, target net, result log
PRED_PATH = os.path.join(args.train_url, 'iqn_pred_net_' + args.games)
TARGET_PATH = os.path.join(args.train_url, 'iqn_target_net_' + args.games)
RESULT_PATH = os.path.join(args.train_url, 'result')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config=config)


class ConvNet():
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size
        # self.g = tf.graph()
        self.action_value, self.tau = self.model()
        vars_list = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name)
        self.saver = tf.train.Saver(vars_list, max_to_keep=5)
        ac_mean = tf.reduce_mean(self.action_value, axis=2)
        self.choose_op = tf.argmax(ac_mean, axis=1)

    def parameters(self):
        param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_name)
        return param

    def model(self):
        with tf.variable_scope(self.model_name):
            self.input_tensor = tf.placeholder(shape=[self.batch_size, 4, 84, 84], dtype=tf.float32)
            input_tensor = self.input_tensor / 128. - 1.
            # input_tensor = tf.placeholder(shape=[BATCH_SIZE, 84, 84, 4], dtype=tf.float32)
            with compat.forward_compatibility_horizon(2019, 5, 1):
                with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, data_format="NCHW"):
                    net = slim.conv2d(input_tensor, num_outputs=32, kernel_size=8, stride=4)
                    net = slim.batch_norm(net, scope="bn1")
                    net = slim.conv2d(net, num_outputs=64, kernel_size=4, stride=2)
                    net = slim.batch_norm(net, scope="bn2")
                    net = slim.conv2d(net, num_outputs=64, kernel_size=3, stride=1)
                    net = slim.batch_norm(net, scope="bn3")
                print("conv_out shape:", net.shape)
                tau = tf.random_uniform(shape=[N_QUANT, 1], minval=0., maxval=1., dtype=tf.float32)
                arange = list(range(0, N_QUANT))
                quants = tf.convert_to_tensor(arange, dtype=tf.float32)
                print("quants shape", quants.shape)
                cos_trans = tf.math.cos(quants * tau * 3.141592)
                cos_trans = tf.expand_dims(cos_trans, axis=2)
                print("cos_trans shape", cos_trans.shape)
                kaiming_init = tf.initializers.variance_scaling(2., distribution="uniform")
                phi = tf.layers.dense(inputs=cos_trans, units=7 * 7 * 64, use_bias=False,
                                      kernel_initializer=kaiming_init)
                print("phi shape", phi.shape)
                # phi = slim.linear(cos_trans, 7 * 7 * 64)
                phi_mean = tf.reduce_mean(phi, axis=1)
                phi_bias = tf.get_variable(shape=[1, 7 * 7 * 64], dtype=tf.float32, name="phi_bias",
                                           initializer=tf.zeros_initializer)
                rand_feat = tf.expand_dims(tf.nn.relu(phi_mean + phi_bias), axis=0)
                net_reshape = tf.expand_dims(tf.reshape(net, shape=[net.shape[0], 3136]), axis=1)
                print("net shape", net_reshape.shape)
                x = net_reshape * rand_feat
                print("x shape", x.shape)
                x = tf.layers.dense(inputs=x, units=512, kernel_initializer=kaiming_init)
                # x = slim.linear(x, 512)
                x = tf.nn.relu(x)
                print("x * rand_feat", x.shape)
                action_value = tf.layers.dense(inputs=x, units=N_ACTIONS, kernel_initializer=kaiming_init)
                # action_value = slim.linear(x, N_ACTIONS)
                print("action_value_shape", action_value.shape)
                action_value = tf.transpose(action_value, [0, 2, 1])  # (m, N_ACTIONS, N_QUANT)
                print(action_value.shape, tau.shape)
        return action_value, tau

    def choose_action(self, s):
        return sess.run(self.choose_op, feed_dict={self.input_tensor: s})

    def __call__(self, input):
        action_value, tau = sess.run([self.action_value, self.tau], feed_dict={self.input_tensor: input})
        return action_value, tau

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.saver.save(sess, os.path.join(path, self.model_name))
        # mox.file.copy_parallel(path, os.path.join(OBS_OUT_PATH, os.path.basename(path)))

    def load(self, path):
        self.saver.restore(sess, os.path.join(path, self.model_name))


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet("pred_net", BATCH_SIZE), ConvNet("target_net", BATCH_SIZE)
        self.IQN_graph()
        self.random_op = tf.random_uniform([N_ENVS], minval=0, maxval=N_ACTIONS, dtype=tf.int32)
        sess.run(tf.global_variables_initializer())
        self.update_op = []
        self.update_choose_op = []
        self.update_rate = tf.placeholder(shape=[], dtype=tf.float32)
        for target_param, pred_param in zip(self.target_net.parameters(), self.pred_net.parameters()):
            self.update_op.append(tf.assign(target_param,
                                            (1.0 - self.update_rate) * target_param + self.update_rate * pred_param))

        self.update_target(1.0)
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0

        # ceate the replay buffer
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)

    # Update target network
    def update_target(self, update_rate):
        # update target network parameters using predcition network
        sess.run(self.update_op, feed_dict={self.update_rate: update_rate})

    def update_choose(self):
        # update target network parameters using predcition network
        sess.run(self.update_choose_op)

    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def Smooth_l1_loss(self, predictions, labels, reduction='none', scope=tf.GraphKeys.LOSSES):
        with tf.variable_scope(scope):
            diff = tf.abs(labels - predictions)
            less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)  # Bool to float32
            smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
            if reduction == 'none':
                return smooth_l1_loss
            else:
                return tf.reduce_mean(smooth_l1_loss)  # 取平均值

    def choose_action(self, x, EPSILON):
        # x:state
        # x = torch.FloatTensor(x)
        # epsilon-greedy
        if np.random.uniform() >= EPSILON:
            # greedy case
            action = self.pred_net.choose_action(x)  # (N_ENVS, N_ACTIONS, N_QUANT)
        else:
            # random exploration case
            action = sess.run(self.random_op)
        return action

    def IQN_graph(self):
        q_eval, q_eval_tau = self.pred_net.action_value, self.pred_net.tau  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        mb_size = BATCH_SIZE
        self.b_a = tf.placeholder(shape=[mb_size], dtype=tf.int32)
        self.b_w = tf.placeholder(shape=[mb_size], dtype=tf.float32)
        self.b_r = tf.placeholder(shape=[mb_size], dtype=tf.float32)
        self.b_d = tf.placeholder(shape=[mb_size], dtype=tf.float32)

        q_eval_split = tf.split(q_eval, mb_size)
        b_a_one_hot = tf.one_hot(self.b_a, depth=N_ACTIONS)
        b_a_split = tf.split(b_a_one_hot, mb_size)
        q_eval = tf.squeeze(tf.stack([tf.matmul(b_a_split[i], tf.squeeze(q_eval_split[i])) for i in range(mb_size)]))
        q_eval = tf.expand_dims(q_eval, axis=2)

        q_next, q_next_tau = self.target_net.action_value, self.target_net.tau  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        q_next_split = tf.split(q_next, mb_size)
        best_actions = tf.argmax(tf.reduce_mean(q_next, axis=2), axis=1)
        best_actions_one_hot = tf.one_hot(best_actions, depth=N_ACTIONS)
        best_actions_split = tf.split(best_actions_one_hot, mb_size)
        q_next = tf.squeeze(tf.stack([tf.matmul(best_actions_split[i], tf.squeeze(q_next_split[i])) for i in range(mb_size)]))
        q_target = tf.expand_dims(self.b_r, axis=1) + GAMMA * (1. - tf.expand_dims(self.b_d, axis=1)) * q_next
        q_target = tf.expand_dims(q_target, axis=1)
        u = q_target - q_eval  # (m, N_QUANT, N_QUANT)
        print("u", u.shape)
        tau = tf.expand_dims(q_eval_tau, axis=0)  # (1, N_QUANT, 1)
        weight = tf.abs(tau - tf.cast(tf.less_equal(u, 0.), dtype=tf.float32))
        loss = self.Smooth_l1_loss(q_eval, q_target, reduction='none')
        #print("loss shape", loss.shape)
        # (m, N_QUANT, N_QUANT)
        loss = tf.reduce_mean(tf.reduce_mean(weight * loss, axis=1), axis=1)
        # calculate importance weighted loss
        self.loss = tf.reduce_mean(self.b_w * loss)
        global_step = tf.train.get_global_step()
        opt = tf.train.AdamOptimizer(learning_rate=LR)
        with tf.name_scope('loss_scale'):
            loss_scale = float(args.loss_scale)
            scaled_grads_and_vars = opt.compute_gradients(self.loss * loss_scale, var_list=self.pred_net.parameters())
            fp32_grads_and_vars = [(tf.cast(g, tf.float32), v) for g, v in scaled_grads_and_vars]
            unscaled_grads_and_vars = [(g / loss_scale, v) for g, v in fp32_grads_and_vars]
            # fp32_grads_and_vars = [(tf.cast(g, tf.float32), v) for g, v in unscaled_grads_and_vars]
            grad_var_list = []
            for g, var in unscaled_grads_and_vars:
                g_and_v = (g, var)
                grad_var_list.append(g_and_v)
        self.train_op = opt.apply_gradients(grad_var_list, global_step=global_step)

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(1.0)
        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        b_w, b_idxes = np.ones_like(b_r), None

        b_s = b_s.astype(np.float32)
        b_a = b_a.astype(np.int32)
        b_r = b_r.astype(np.float32)
        b_s_ = b_s_.astype(np.float32)
        b_d = b_d.astype(np.float32)
        b_w = b_w.astype(np.float32)

        feed_dict = {self.target_net.input_tensor: b_s_, self.pred_net.input_tensor: b_s, self.b_a: b_a,
                     self.b_w: b_w, self.b_r: b_r, self.b_d: b_d}
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        #self.update_choose()
        return loss


def train(epsilon, model, restart_step=0):
    print('Collecting experience...')
    # episode step for accumulate reward
    epinfobuf = deque(maxlen=100)
    # check learning time
    start_time = time.time()

    # env reset
    s = np.array(env.reset())
    for step in range(restart_step + 1, int(STEP_NUM // N_ENVS) + 1):
        a = model.choose_action(s, epsilon)
        # print('a',a)

        # take action and get next state
        s_, r, done, infos = env.step(a)
        # log arrange
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfobuf.append(maybeepinfo)
        s_ = np.array(s_)

        # clip rewards for numerical stability
        clip_r = np.sign(r)

        # store the transition
        for i in range(N_ENVS):
            model.store_transition(s[i], a[i], clip_r[i], s_[i], done[i])

        # annealing the epsilon(exploration strategy)
        if step <= int(1e+4):
            # linear annealing to 0.9 until million step
            epsilon = EPSILON - 0.9 / 1e+4 * step
        elif step <= int(2e+4):
            # else:
            # linear annealing to 0.99 until the end
            epsilon = 0.1 - 0.09 / 1e+4 * (step - 1e+4)
        else:
            epsilon = 0.01

        # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
        if (LEARN_START <= model.memory_counter) and (model.memory_counter % LEARN_FREQ == 0):
            loss = model.learn()

        # print log and save
        if step % SAVE_FREQ == 0:
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return
            mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
            if math.isnan(mean_100_ep_return):
                mean_100_ep_return = -1000.
            result.append(mean_100_ep_return)
            # print log
            print('Step: ', step * N_ENVS,
                  '| EPS: ', round(epsilon, 3),
                  # '| Loss: ', loss,
                  '| return: ', mean_100_ep_return,
                  '| Time:', time_interval,
                  '| max_r:', max(result))
            # save model
            if step % (SAVE_FREQ * 10) == 0:
                model.save_model()
                pkl_file = open(os.path.join(RESULT_PATH, 'iqn_result_' + args.games + '.pkl'), 'wb')
                pickle.dump(np.array(result), pkl_file)
                pkl_file.close()
                # mox.file.copy_parallel(RESULT_PATH, OBS_RESULT_PATH)
                print("save done")
        s = s_

        if RENDERING:
            env.render()
    print("The training is done!")


def eval(model):
    env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(1)])
    _100_ep_return = []
    print("start evaluate")
    # for i in range(5):
    start = time.time()
    s = np.array(env.reset())
    while True:
        action = model.choose_action(s)
        s_, r, done, infos = env.step(action)
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                _100_ep_return.append(maybeepinfo['r'])
                print("epoch {} return: {}".format(len(_100_ep_return), maybeepinfo['r']))
        if len(_100_ep_return) == 100:
            break
        s = s_
    print("mean_100_ep_return:", round(np.mean(_100_ep_return), 2))
    return


if __name__ == "__main__":
    dqn = DQN()
    step = 0
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    if not os.path.exists(PRED_PATH):
        os.mkdir(PRED_PATH)
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)

    if LOAD and os.path.exists(PRED_PATH) and os.path.exists(TARGET_PATH):
        dqn.load_model()
        pkl_file = open(os.path.join(RESULT_PATH, 'iqn_result_' + args.games + '.pkl'), 'rb')
        result = list(pickle.load(pkl_file))
        pkl_file.close()
        step = int(len(result) * SAVE_FREQ)
        print("result num:", len(result))
        print('Load complete!')
    else:
        result = []
        print('Initialize results!')

    if args.mode == 'train':
        train(EPSILON, dqn, step)
    elif args.mode == 'evaluate':
        dqn.load_model()
        eval_net = ConvNet("eval_choose", 1)
        update_choose_op = []
        for choose_parm, pred_param in zip(eval_net.parameters(), dqn.pred_net.parameters()):
            update_choose_op.append(tf.assign(choose_parm, pred_param))
        sess.run(update_choose_op)
        eval(eval_net)

