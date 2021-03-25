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
"""server agent for dqn algorithm, usage for self-play task."""

from xt.agent.dqn.cartpole_dqn import CartpoleDqn
from xt.framework.register import Registers


@Registers.agent.register
class ServerAgent(CartpoleDqn):
    """ Server Agent with dqn algorithm."""

    def run_one_episode(self, use_explore, need_collect):
        """
        run episode forever.
        :param use_explore:
        :param need_collect: if collect the total transition of each episode.
        :return:
        """
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.reset_env(self.id)
        while True:
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                break

        return self.get_trajectory()

    # def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
    #     if info.get("reset"):
    #         return self.transition_data
    #     super().handle_env_feedback(
    #         next_raw_state, reward, done, info, use_explore
    #     )
