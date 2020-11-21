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
"""reinforce algorithm implemented with pytorch."""
import os
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.utils as utils
import torch.optim as optim
from torch.autograd import Variable
from xt.algorithm import MODEL_PREFIX, Algorithm
from xt.algorithm.reinforce.default_config import LR
from xt.framework.register import Registers
from xt.util.common import import_config


@Registers.algorithm.register
class Reinforce(Algorithm):
    """reinforce algorithm"""

    def __init__(self, model_info, alg_config, **kwargs):
        import_config(globals(), alg_config)
        super(Reinforce, self).__init__(
            alg_name="reinforce", model_info=model_info["actor"], alg_config=alg_config
        )

        self.optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.actor.train()
        self.entropies = []
        self.log_probs = []
        self.rewards = []
        self.state = []
        self.action = []
        self.async_flag = False

    def predict(self, state):
        state = torch.Tensor([state])
        probs = self.actor(Variable(state))
        action = probs.multinomial(1).data
        action = action[0]
        action = action.cpu()
        action = action.numpy()[0]
        return action

    def prepare_data(self, train_data, **kwargs):
        data_len = len(train_data["done"])
        for index in range(data_len):
            s_t, action, r_t = (
                train_data["cur_state"][index],
                train_data["action"][index],
                train_data["reward"][index],
            )
            self.rewards.append([])
            self.state.append([])
            self.action.append([])
            self.rewards[index].append(r_t)
            self.state[index].append(s_t)
            self.action[index].append(action)

    def train(self, **kwargs):
        """train process"""
        loss = []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                action = [self.action[i][j]]
                action = [action]
                action = np.array(action)
                action = torch.from_numpy(action)
                state = torch.Tensor([self.state[i][j]])

                probs = self.actor(Variable(state))
                prob = probs[:, action[0, 0]].view(1, -1)
                log_prob = prob.log()
                entropy = -(probs * probs.log()).sum()
                self.entropies.append(entropy)
                self.log_probs.append(log_prob)

            r__ = torch.zeros(1, 1)
            loss.append(0)
            gamma = 0.99
            for j in reversed(range(len(self.rewards[i]))):
                r__ = gamma * r__ + self.rewards[i][j]
                loss[i] = (
                    loss[i]
                    - (
                        self.log_probs[j] * (Variable(r__).expand_as(self.log_probs[j]))
                    ).sum()
                    - (0.0001 * self.entropies[j]).sum()
                )

            self.log_probs = []
            self.entropies = []
            if not self.rewards[i]:
                loss[i] = loss[i]
            else:
                loss[i] = loss[i] / len(self.rewards[i])

        loss_avg = 0
        for item in loss:
            loss_avg += item
        loss_avg = loss_avg / len(loss)
        self.optimizer.zero_grad()
        loss_avg.backward()
        utils.clip_grad_norm(self.actor.parameters(), 40)
        self.optimizer.step()

        self.rewards = []
        self.state = []
        self.action = []

    def save(self, model_path, model_index):
        """rewrite save func with pytorch"""
        save_name = "{}_{}.pkl".format(MODEL_PREFIX, str(model_index))
        torch.save(self.actor.state_dict(), os.path.join(model_path, save_name))
        return [save_name]

    def restore(self, model_name, model_weights=None):
        """rewrite restore func with pytorch"""
        if model_name is None:
            for key, ndarr in model_weights.items():
                model_weights[key] = torch.Tensor(ndarr)
            self.actor.load_state_dict(model_weights)
        else:
            # model_name = model_name[0]
            torch_load = torch.load(model_name)
            self.actor.load_state_dict(torch_load)
        # print("load_model")

    def get_weights(self):
        """rewrite get weights func with pytorch"""
        dict_data = self.actor.state_dict()
        numpy_state_dict = OrderedDict()
        for key, tensor in dict_data.items():
            numpy_state_dict[key] = tensor.cpu().numpy()
        return numpy_state_dict
