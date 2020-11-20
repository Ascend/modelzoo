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
"""impala algorithm"""
import os

import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.impala.default_config import BATCH_SIZE, GAMMA
from xt.framework.register import Registers
from xt.model.tf_compat import loss_to_val
from xt.util.common import import_config


@Registers.algorithm.register
class IMPALA(Algorithm):
    """IMPALA algorithm"""
    def __init__(self, model_info, alg_config, **kwargs):
        import_config(globals(), alg_config)
        super(IMPALA, self).__init__(alg_name="impala",
                                     model_info=model_info["actor"],
                                     alg_config=alg_config)

        self.dummy_action, self.dummy_value = (
            np.zeros((1, self.action_dim)),
            np.zeros((1, 1)),
        )
        self.state = None
        self.target_value = None
        self.pg_adv = None
        self.action = None
        self.async_flag = False

    def train(self, **kwargs):
        """refer to alg"""
        state = self.state
        pg_adv = self.pg_adv
        target_value = self.target_value
        action_matrix = self.action

        nbatch = len(self.state)
        count = (nbatch + BATCH_SIZE - 1) // BATCH_SIZE
        # print("state_size: {}, train_count: {}".format(np.shape(self.state), count))
        loss_list = []
        for start in range(count):
            start_index = start * BATCH_SIZE
            env_index = start_index + BATCH_SIZE
            state_fit = state[start_index:env_index]
            pg_adv_fit = pg_adv[start_index:env_index]
            value_fit = target_value[start_index:env_index]
            action_matrix_fit = action_matrix[start_index:env_index]

            # print([np.shape(_d) for _d in [action_matrix_fit, value_fit]])

            actor_loss = self.actor.train([state_fit, pg_adv_fit], [action_matrix_fit, value_fit])
            loss_list.append(loss_to_val(actor_loss))
        self.state = None
        return np.mean(loss_list)

    def get_grad(self):
        state = self.state
        pg_adv = self.pg_adv
        target_value = self.target_value
        action = self.action

        grad_list = []
        # BATCH_SIZE = 32
        b_count = (state.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
        # print(b_count, id_)
        for start in range(b_count):
            start_index = start * BATCH_SIZE
            env_index = start_index + BATCH_SIZE  # np make end actual
            state_fit = state[start_index:env_index]
            pg_adv_fit = pg_adv[start_index:env_index]
            target_value_fit = target_value[start_index:env_index]
            action_fit = action[start_index:env_index]
            grad_input = [state_fit, pg_adv_fit, action_fit, target_value_fit]
            grad = self.actor.get_grad(grad_input)
            grad_list.append(grad)

        grad = []
        for i in range(len(grad_list)):
            for j in range(len(grad_list[i])):
                if i == 0:
                    grad.append(grad_list[i][j])
                else:
                    grad[j] += grad_list[i][j]

        self.state = None
        return grad

    def apply_grad(self, grad):
        grads_ave = {}
        for i in range(len(self.actor.grads_holder)):
            k = self.actor.grads_holder[i][0]
            if k is not None:
                grads_ave[k] = grad[i]
        self.actor.sess.run(self.actor.train_op, feed_dict=grads_ave)

    def save(self, model_path, model_index):
        """refer to alg"""
        actor_name = "actor" + str(model_index).zfill(5)
        actor_name = self.actor.save_model(os.path.join(model_path, actor_name))
        actor_name = actor_name.split("/")[-1]

        return [actor_name]

    def prepare_data(self, train_data, **kwargs):
        """prepare the data for impala algorithm"""
        target_value, pg_adv, state, action = self._data_proc(train_data)
        if self.state is None:
            self.state = state
            self.target_value = target_value
            self.pg_adv = pg_adv
            self.action = action
        else:  # train with prepare many times data
            self.state = np.vstack((self.state, state))
            self.target_value = np.vstack((self.target_value, target_value))
            self.pg_adv = np.vstack((self.pg_adv, pg_adv))
            self.action = np.vstack((self.action, action))

    def predict(self, state):
        """refer to alg"""
        state = state.reshape((1, ) + state.shape)
        dummp_value = np.zeros((1, 1))
        pred = self.actor.predict([state, dummp_value])

        return pred

    def _data_proc(self, episode_data):
        """data process for impala"""
        dic = dict()
        # dic['states'] = [e[0] for e in episode_data]
        dic["states"] = episode_data["cur_state"]

        # dic['actions'] = np.asarray([e[1] for e in episode_data])
        dic["actions"] = np.asarray(episode_data["real_action"])

        dic["rewards"] = np.asarray(episode_data["reward"])
        dic["rewards"] = dic["rewards"].reshape((dic["rewards"].shape[0], 1))
        # dic["new_states"] = episode_data["next_state"]
        dic["dones"] = np.asarray(episode_data["done"])
        dic["dones"] = dic["dones"].reshape((dic["dones"].shape[0], 1))
        dic["pred_a"] = np.asarray(episode_data["action"])
        dic["states"].append(episode_data["last_state"])
        # print(states[-1])
        dic["states"] = np.asarray(dic["states"])

        # convert action to oneHot
        dic["action_matrix"] = np.eye(self.action_dim)[dic["actions"].reshape(-1)]
        dic["outputs"] = self.actor.predict(
            [dic["states"], np.zeros((dic["states"].shape[0], 1))])
        dic["probs"] = dic["outputs"][0]
        dic["values"] = dic["outputs"][1]
        dic["value"] = dic["values"][:-1]
        dic["value_next"] = dic["values"][1:]
        target_action = dic["probs"][:-1]
        discounts = ~dic["dones"] * GAMMA

        behaviour_logp = self._logp(dic["pred_a"], dic["action_matrix"])
        target_logp = self._logp(target_action, dic["action_matrix"])
        radio = np.exp(target_logp - behaviour_logp)
        radio = np.minimum(radio, 1.0)
        radio = radio.reshape((radio.shape[0], 1))
        deltas = radio * (dic["rewards"] + discounts * dic["value_next"] - dic["value"])

        adv = deltas
        for j in range(len(adv) - 2, -1, -1):
            adv[j] += adv[j + 1] * discounts[j + 1] * radio[j + 1]

        target_value = dic["value"] + adv
        target_value_next = target_value[1:]
        target_value_next = np.r_[
            target_value_next,
            dic["value_next"][-1].reshape((1, ) + dic["value_next"][-1].shape), ]
        pg_adv = radio * (dic["rewards"] + discounts * target_value_next - dic["value"])

        return (
            target_value,
            pg_adv,
            np.asarray(dic["states"][:-1]),
            dic["action_matrix"],
        )

    @staticmethod
    def _logp(prob, action):
        """to be filled"""
        action_prob = np.sum(prob * action, axis=-1)
        return np.log(action_prob + 1e-10)
