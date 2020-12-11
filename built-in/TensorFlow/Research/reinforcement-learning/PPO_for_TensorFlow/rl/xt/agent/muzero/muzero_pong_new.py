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
import numpy as np
from xt.agent.muzero.muzero import Muzero
from xt.agent.muzero.mcts import Mcts
from xt.agent.muzero.default_config import NUM_SIMULATIONS, GAMMA, TD_STEP
from xt.framework.register import Registers

@Registers.agent
class MuzeroPongNew(Muzero):
    """ Agent with Muzero algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS
        self.explore_count = 0

    def infer_action(self, state, use_explore):
        """
        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        state = state.astype('uint8')
        if True:
            mcts = Mcts(self, state)
            if use_explore:
                mcts.add_exploration_noise(mcts.root)
            # policy, value = self.alg.actor.ppo_infer(state)
            # policy = list(policy[0])
            # value = value[0][0]

            mcts.run_mcts()
            action = mcts.select_action()

            # node = mcts.root
            # visit_counts = [child.visit_count for child in node.children.values()]

            # print("root policy", mcts.policy, visit_counts, policy)
            # action = np.random.choice(self.alg.action_dim, p=np.nan_to_num(mcts.policy))
            # action = np.random.choice(self.alg.action_dim, p=np.nan_to_num(policy))
            self.transition_data.update({"cur_state": state, "action": action})
            self.transition_data.update(mcts.get_info())
        else:
            policy, value = self.alg.actor.ppo_infer(state)
            policy = list(policy[0])
            value = value[0][0]
            # print(policy, value)
            action = np.random.choice(self.alg.action_dim, p=np.nan_to_num(policy))
            self.transition_data.update({"cur_state": state, "action": action})
            self.transition_data.update({"child_visits": policy, "root_value": value})

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        # if reward > 0:
        #     print("real_reard", reward)
        #
        # if done:
        #     print("done", info)
        next_raw_state = next_raw_state.astype('uint8')
        super().handle_env_feedback(next_raw_state, reward, done, info, use_explore)

        return self.transition_data

    def run_one_episode(self, use_explore, need_collect):
        """
        In each episode, do interaction with max steps.
        :param use_explore:
        :param need_collect: if collect the total transition of each episode.
        :return:
        """
        # clear the old trajectory data

        self.clear_trajectory()
        state = self.env.get_init_state(self.id)

        for _ in range(self.max_step):
            self.clear_transition()

            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                self.env.reset()
                state = self.env.get_init_state(self.id)
                # break
        traj = self.get_trajectory()
        return traj

    def get_trajectory(self):
        self.data_proc()
        return super().get_trajectory()

    def data_proc(self):
        traj = self.trajectory
        value = traj["root_value"]
        reward = traj["reward"]
        dones = np.asarray(traj["done"])
        discounts = ~dones * GAMMA

        target_value = [reward[-1]] * len(reward)
        for i in range(len(reward) - 1):
            end_index = min(i + TD_STEP, len(reward) - 1)
            sum_value = value[end_index]

            for j in range(i , end_index)[::-1]:
                sum_value = reward[j] +  discounts[j] * sum_value

            target_value[i] = sum_value

        self.trajectory["target_value"] = target_value
        # print(reward)
        # print(value)
        # print(target_value)
        # print(traj)
        self.explore_count += 1
        if self.explore_count % 5000000 == 0:
            print(value)
            print(reward)
            print(traj["action"])
            print(traj["child_visits"])
