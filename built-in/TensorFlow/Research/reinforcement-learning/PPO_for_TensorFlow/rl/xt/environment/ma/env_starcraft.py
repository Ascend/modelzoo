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
use multiagent environment from smac.
```
def get_env_info(self):
    env_info = {"state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit}
    return env_info
```
"""
import os
import sys

from absl import logging
from smac.env import MultiAgentEnv, StarCraft2Env
from xt.environment.environment import Environment
from xt.framework.register import Registers

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


@Registers.env
class StarCraft2Xt(Environment):
    """make starcraft II simulation into xingtian's environment"""
    def init_env(self, env_info):
        logging.debug("init env with: {}".format(env_info))
        print(env_info)
        sys.stdout.flush()
        _info = env_info.copy()
        if "agent_num" in _info.keys():
            _info.pop("agent_num")
        return StarCraft2Env(**_info)

    def reset(self):
        """
        reset the environment. starcraft env need get obs & global status

        :return: None
        """
        self.env.reset()
        return None

    def step(self, action, agent_index=0):
        """simplest step in starcraft."""
        reward, done, info = self.env.step(action)
        return reward, done, info

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_obs(self):
        return self.env.get_obs()

    def get_env_info(self):
        """return environment's basic information"""
        self.reset()
        env_attr = self.env.get_env_info()
        env_attr.update(
            {"api_type": self.api_type,}
        )
        # update the agent ids, will used in the weights map.
        # default work well with the sumo multi-agents
        # starcraft multi-agents consider as a batch agents.
        env_attr.update({"agent_ids": [0,]})

        return env_attr
