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
"""MuZero agent."""
import numpy as np

from xt.agent.agent import Agent
from xt.agent.muzero.default_config import NUM_SIMULATIONS
from xt.agent.muzero.mcts import Mcts
from xt.framework.register import Registers
from xt.util.common import import_config

@Registers.agent.register
class Muzero(Agent):
    """ Agent with Muzero algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        import_config(globals(), agent_config)
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS

    def infer_action(self, state, use_explore):
        """
        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        # state = state.astype('int8')
        mcts = Mcts(self, state)
        if use_explore:
            mcts.add_exploration_noise(mcts.root)

        mcts.run_mcts()
        action = mcts.select_action()
        # print("action", action)

        self.transition_data.update({"cur_state": state, "action": action})
        self.transition_data.update(mcts.get_info())

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        info.update({'eval_reward': reward})
        # done = info.get('real_done', done)

        self.transition_data.update({
            "reward": np.sign(reward) if use_explore else reward,
            "done": done,
            "info": info
        })

        return self.transition_data

    def sync_model(self):
        # model_name = self.recv_explorer.recv(name=None, block=True)
        model_name = "none"
        try:
            while True:
                model_name = self.recv_explorer.recv(name=None, block=False)
        except:
            pass
        return model_name
