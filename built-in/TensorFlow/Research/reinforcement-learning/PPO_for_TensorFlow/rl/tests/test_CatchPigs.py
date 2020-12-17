# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
test with unittest.
usages:
    cd xt
    python -m unittest tests/test_CatchPigs.py
"""
import random
import unittest
from time import sleep

import numpy as np

from xt.environment.ma.env_CatchPigs import EnvCatchPigs


class TestStringMethods(unittest.TestCase):
    def test_random_agent(self):
        env = EnvCatchPigs(7, True)
        max_iter = 10000
        for i in range(max_iter):
            action1 = random.randint(0, 4)
            action2 = random.randint(0, 4)
            action_list = [action1, action2]
            obs_list = env.get_obs()
            obs1 = obs_list[0]
            obs2 = obs_list[1]
            # print(obs_list)
            # print("obs shape: ", np.shape(obs1), np.shape(obs2))
            # print("iter={}, {}, {}, {}, {}, {}, action: {}, {}".format(
            #     i,
            #     env.agt1_pos,
            #     env.agt2_pos,
            #     env.pig_pos,
            #     env.agt1_ori,
            #     env.agt2_ori,
            #     action1,
            #     action2,
            # ))
            # env.render()
            # sleep(2)
            reward_list, done = env.step(action_list)
            # print(done)
            if done:
                print("{}, {}".format(i, done))
            # env.plot_scene()
            if reward_list[0] > 0:
                print(reward_list)
                print("iter= ", i)
                print("agent 1 finds goal")
                break


if __name__ == "__main__":
    unittest.main()
