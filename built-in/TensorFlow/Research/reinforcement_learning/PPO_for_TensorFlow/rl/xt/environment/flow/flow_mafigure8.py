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
import copy
import json
import math
import random
import sys
import time

import numpy as np
from xt.environment.environment import Environment
from xt.framework.register import Registers

try:
    from flow.controllers import (ContinuousRouter, IDMController, RLController,
                                SimCarFollowingController)

    from flow.core.params import (EnvParams, InitialConfig, NetParams,
                                SumoCarFollowingParams, SumoParams,
                                VehicleParams)
    from flow.envs import AccelEnv
    from flow.envs.multiagent import MultiAgentAccelPOEnv, MultiAgentEightEnv
    from flow.networks import FigureEightNetwork
    from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
    from flow.utils.registry import make_create_env
except (ModuleNotFoundError, ImportError) as err:
    print("import error")


@Registers.env
class MaEnvFigure8(Environment):
    """kybersim environment for lanechange case"""
    def init_env(self, env_info):
        """
        cEnvFigure8reate a flow environment instance

        :param: the config information of environment
        :return: the instance of environment
        """
        self.init_state = None
        # update the agents number and env api type.
        self.n_agents = 10
        self.api_type = "unified"

        # We place one autonomous vehicle and 13 human-driven vehicles in the network
        vehicles = VehicleParams()
        vehicles.add(
            veh_id='rl',
            # initial_speed=5,
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                # speed_mode="aggressive",
                decel=1.5,
            ),
            num_vehicles=10)

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim = SumoParams(
            sim_step=0.1,
            render=False,
            restart_instance=True,
        )

        # environment related parameters (see flow.core.params.EnvParams)
        env_params = EnvParams(
            horizon=1000,
            additional_params={
                'target_velocity': 20,
                'max_accel': 3,
                'max_decel': 3,
                'sort_vehicles': False
            },
        )

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net = NetParams(additional_params=ADDITIONAL_NET_PARAMS.copy(), )

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)

        network = FigureEightNetwork(name='multiagent_figure_eight',
                                     vehicles=vehicles,
                                     net_params=net)

        # parameters specifying the positioning of vehicles upon initialization/
        env = MultiAgentEightEnv(env_params, sim, network)

        return env

    def reset(self):
        """
        reset the environment, if there are illegal data in observation
        then use old data

        :param reset_arg: reset scene information
        :return: the observation of environment
        """

        state = self.env.reset()
        self.init_state = state
        return state

    def step(self, actions, agent_index=0):
        """
        send lanechange cmd to kybersim

        :param action: action （0-2）
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """

        state_, reward, done, _ = self.env.step(actions)
        return state_, reward, done, _

    def get_init_state(self, agent_index=0):
        """
        get reset observation of one agent.

        :param agent_index: the index of agent
        :return: the reset observation of agent
        """
        return self.init_state

    def close(self):
        """
        close environment
        """
        try:  # do some thing you need
            self.env.terminate()
        except AttributeError:
            print("please complete your env close function")

    def get_env_info(self):
        """rewrite environment's basic information, indicate multi-agents"""
        self.reset()
        env_info = {
            "n_agents": self.n_agents,
            "api_type": self.api_type,
        }
        # update the agent ids, will used in the weights map.
        # default work well with the sumo multi-agents
        agent_ids = list(
            self.get_init_state().keys())  # if self.n_agents > 1 else [0]
        env_info.update({"agent_ids": agent_ids})

        return env_info
