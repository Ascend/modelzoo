# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""sumo env for simulation."""
import os
from collections import namedtuple

from xt.environment.environment import Environment
from xt.environment.sumo.traffic_env import TrafficEnv
from xt.framework.register import Registers

EgoVehicle = namedtuple(
    "EgoVehicle",
    [
        "vehID",
        "routeID",
        "typeID",
        "departPos",
        "departSpeed",
        "arrivalPos",
        "departLane",
    ],
)
ENV_CONFIG = {
    "sumo_cfg_file": os.path.join("./xt/environment/sumo/", "sumo_env", "sumo.sumocfg"),
    "ego_vehicle_init": [
        EgoVehicle("ego_car", "left", "veh_passenger", 170.0, 0.0, 10.0, "1"),
        EgoVehicle("ego_car", "mid", "veh_passenger", 170.0, 0.0, 10.0, "0"),
        EgoVehicle("ego_car", "right", "veh_passenger", 170.0, 0.0, 10.0, "0"),
    ],
    "mode": "cli",
    "step_length": "0.1",
    "simulation_end": 31536000,
    "discrete_actions": False,
    "squash_action_logits": False,
}

@Registers.env
class SumoEnv(Environment):
    """
     environment basic class
    """

    # def __init__(self, env_info, **kwargs):
    #     """
    #     :param env_info: the config info of environment
    #     """
    #     Environment.__init__(self, env_info, **kwargs)
    #     self.env = self.init_env(env_info)
    #     self.init_state = None

    def init_env(self, env_info):
        """
        create an environment instance

        :param: the config information of environment
        :return: the instance of environment
        """
        env = TrafficEnv(config=env_info)
        self.init_state = None
        return env

    def reset(self):
        """
        reset the environment.

        :return: the observation of environment
        """
        state = self.env.reset()
        self.init_state = state

        return state

    def step(self, action, agent_index=0):
        """
        send action  to running agent in this environment.

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward
        """
        return self.env.step(action)

    # def get_init_state(self, agent_index=0):
    #     """
    #     get reset observation of one agent.
    #
    #     :param agent_index: the index of agent
    #     :return: the reset observation of agent
    #     """
    #     return self.init_state

    def close(self):
        """
        close environment
        """
        try:  # do some thing you need
            # self.env.close()
            pass
        except AttributeError:
            print("please complete your env close function")


# def create_env(env_info, **kwargs):
#     """ create environment interface """
#     return SumoEnv(env_info, **kwargs)
