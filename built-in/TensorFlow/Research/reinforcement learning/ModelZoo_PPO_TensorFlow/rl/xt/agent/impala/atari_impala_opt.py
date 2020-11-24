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
"""Vector multi-environment in Atari agent for impala algorithm."""

import numpy as np
from collections import defaultdict, deque

from xt.agent.ppo.cartpole_ppo import CartpolePpo
from xt.framework.comm.message import message, set_msg_info
from xt.framework.register import Registers


@Registers.agent
class AtariImpalaOpt(CartpolePpo):
    """Atari Agent with Impala algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        self.vector_env_size = kwargs.pop("vector_env_size")

        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True  # to keep max sequence length in explorer.
        self.next_logit = None
        self.broadcast_weights_interval = 2
        self.sync_weights_count = 0

        # vector environment will auto reset in step
        self.transition_data["done"] = False
        self.sample_vector = dict()
        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id] = defaultdict(list)

        self.reward_track = deque(
            maxlen=self.vector_env_size*self.broadcast_weights_interval)
        self.reward_per_env = defaultdict(float)

    def get_explore_mean_reward(self):
        """calculate explore reward among limited trajectory"""
        return np.nan if not self.reward_track else np.nanmean(self.reward_track)

    def infer_action(self, state, use_explore):
        """
        Infer an action with the `state`
        :param state:
        :param use_explore:
        :return: action value
        """
        if self.next_state is None:
            s_t = (np.array(state, dtype="int16")-128).astype("int8")
            predict_val = self.alg.predict(s_t)

            logit = predict_val[0]
            value = predict_val[1]
            action = predict_val[2]
        else:
            s_t = self.next_state
            logit = self.next_logit
            action = self.next_action
            value = self.next_value

        # update transition data
        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id]["cur_state"].append(s_t[env_id])
            self.sample_vector[env_id]["logit"].append(logit[env_id])
            self.sample_vector[env_id]["action"].append(action[env_id])

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        """handle next state, reward and info"""
        next_state = (np.array(next_raw_state, dtype="int16")-128).astype("int8")
        predict_val = self.alg.predict(next_state)

        self.next_logit = predict_val[0]
        self.next_value = predict_val[1]
        self.next_action = predict_val[2]

        self.next_state = next_state

        for env_id in range(self.vector_env_size):
            info[env_id].update({'eval_reward': reward[env_id]})
            self.reward_per_env[env_id] += reward[env_id]

            if info[env_id].get('real_done'):  # real done
                self.reward_track.append(self.reward_per_env[env_id])
                self.reward_per_env[env_id] = 0

        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id]["reward"].append(reward[env_id])
            self.sample_vector[env_id]["done"].append(done[env_id])
            self.sample_vector[env_id]["info"].append(info[env_id])

        return self.transition_data

    def get_trajectory(self):
        for env_id in range(self.vector_env_size):
            for _data_key in ("cur_state", "logit", "action", "reward", "done", "info"):
                self.trajectory[_data_key].extend(self.sample_vector[env_id][_data_key])

        # merge data into env_num * seq_len
        for _data_key in self.trajectory:
            self.trajectory[_data_key] = np.stack(self.trajectory[_data_key])

        self.trajectory["action"].astype(np.int32)

        trajectory = message(self.trajectory.copy())
        set_msg_info(trajectory, agent_id=self.id)
        return trajectory

    def sync_model(self):
        """block wait one [new] model when sync need"""
        model_name = "none"
        self.sync_weights_count += 1
        if self.sync_weights_count >= self.broadcast_weights_interval:
            model_name = self.recv_explorer.recv(block=True)
            self.sync_weights_count = 0

            try:
                while True:
                    model_name = self.recv_explorer.recv(block=False)
            except:
                pass

        return model_name

    def reset(self):
        """clear the sample_vector buffer"""
        self.sample_vector = dict()
        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id] = defaultdict(list)

    def add_to_trajectory(self, transition_data):
        pass
