# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""double dqn algorithm."""
import numpy as np

from xt.algorithm.dqn.default_config import BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ, TAU, UPDATE_FREQ
from xt.algorithm.dqn.dqn import DQN
from xt.framework.register import Registers


@Registers.algorithm
class DDQN(DQN):
    """double DQN algorithm"""
    def __init__(self, model_info, alg_config, **kwargs):
        self.step = 0
        # self.buff = ReplayBuffer(BUFFER_SIZE)
        self.gamma = GAMMA
        self.tau = TAU
        self.batch_size = BATCH_SIZE
        self.update_freq = UPDATE_FREQ

        super(DDQN, self).__init__(model_info, alg_config)

    def predict(self, state):
        """Predict actions by policy subnet
        """
        action = self.actor.predict(state)
        a_int = int(np.argmax(action, axis=1)[0])
        return a_int

    def train(self, **kwargs):
        """DDQN training"""
        # sample mini-batch
        batch = self.buff.get_batch(self.batch_size)
        s_batch = np.asarray([data[0] for data in batch])
        r_batch = np.asarray([data[2] for data in batch])
        s1_batch = np.asarray([data[3] for data in batch])
        a_q = self.actor.predict(s1_batch)
        a_pred = np.argmax(a_q, axis=1)
        a1_batch = np.zeros((len(batch), self.action_dim))
        for i in range(len(batch)):
            a1_batch[i][a_pred[i]] = 1.0

        a_tq = self.target_actor.predict(s1_batch)
        q_t = np.sum(np.multiply(a_tq, a1_batch), axis=1)
        q1_batch = np.reshape(q_t, (-1))
        target_q_batch = r_batch + self.gamma * q1_batch
        # print("reward =", np.sum(r_batch))
        # update main network
        for i in range(len(batch)):
            a1_batch[i][a_pred[i]] = target_q_batch[i]

        s_batch = np.reshape(s_batch, (-1, s_batch.shape[-1]))
        loss = self.actor.train(s_batch, a1_batch)

        # update target network
        self.train_count += 1
        if self.train_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()
        return loss
