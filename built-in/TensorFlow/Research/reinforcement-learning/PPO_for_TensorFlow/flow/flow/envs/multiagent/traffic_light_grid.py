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
"""Multi-agent environments for scenarios with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
# from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
# from flow.multiagent_envs.multiagent_env import MultiEnv
from flow.envs.multiagent import MultiEnv
from collections import defaultdict

import time
import copy
import math

ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
    # maximum acceleration of autonomous vehicles
    'max_accel': 5,
    # maximum deceleration of autonomous vehicles
    'max_decel': 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 5
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1
STATE_DIM = 4
NUM_VEHICLE = 3
X_LENGTH = 60
Y_LENGTH = 60
MAX_ANGLE = 360
MAX_SPEED = 10
LEAD_HEAD = 15
TTC_HEAD = 2
INFLOW_EDGE = {'bot0_0', 'right0_0', 'top0_1', 'left1_0'}
OUTFLOW_EDGE = {'bot0_1', 'top0_0', 'left0_0', 'right1_0'}

GOAL = 29.5

class MultiTrafficLightGridPOEnv(TrafficLightGridPOEnv, MultiEnv):
    """Multiagent shared model version of PO_TrafficLightGridEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        # super().__init__(env_params, sim_params, scenario, simulator)

        # for p in ADDITIONAL_ENV_PARAMS.keys():
        #     if p not in env_params.additional_params:
        #         raise KeyError(
        #             'Environment parameter "{}" not supplied'.format(p))

        # # number of nearest lights to observe, defaults to 4
        # self.num_local_lights = env_params.additional_params.get(
        #     "num_local_lights", 4)

        # # number of nearest edges to observe, defaults to 4
        # self.num_local_edges = env_params.additional_params.get(
        #     "num_local_edges", 4)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        self.alpha = env_params.additional_params['alpha']
        self.accel = defaultdict(int)
        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, local edge information, and traffic
        light state.
        """
        # tl_box = Box(
        #     low=0.,
        #     high=1,
        #     shape=(3 * 4 * self.num_observed +
        #            2 * self.num_local_edges +
        #            3 * (1 + self.num_local_lights),
        #            ),
        #     dtype=np.float32)
        # return tl_box
        """See class definition."""
        return Box(low=-2, high=2, shape=(8, ), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        # if self.discrete:
        #     return Discrete(2)
        # else:
        #     return Box(
        #         low=-1,
        #         high=1,
        #         shape=(1,),
        #         dtype=np.float32)
        return Box(
            # low=-np.abs(self.env_params.additional_params['max_decel']),
            # high=self.env_params.additional_params['max_accel'],
            low=-1,
            high=1,
            shape=(1,),  # (4,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            # print(rl_id, this_speed)
            # log = open("%s" % rl_id, "a")
            # log.write(str(this_speed) + "\n")
            # log.close
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)
            ego_orientation = self.k.vehicle.get_orientation(rl_id)
            rl_orientation = ego_orientation[2]
            # this_acceleration = self.k.vehicle.get_acceleration(rl_id)
            # log_file = open("id_speed_acc.txt", "a")
            # log_file.write(rl_id + '\t' + str(this_speed) + '\n')
            # log_file.close()
            # print(rl_id, this_speed, this_acceleration)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = MAX_SPEED
                lead_head = X_LENGTH
                lead_orientation = rl_orientation
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                # lead_head = self.k.vehicle.get_headway(rl_id)
                lead_position = self.k.vehicle.get_orientation(lead_id)
                lead_orientation = lead_position[2]
                lead_x = ego_orientation[0] - lead_position[0]
                lead_y = ego_orientation[1] - lead_position[1]
                lead_head  = np.sqrt(pow(lead_x, 2) + pow(lead_y, 2))

            if follower in ["", None]:
                # in case follower is not visible
                follower_speed = 0
                follower_head = X_LENGTH
                follower_orientation = rl_orientation
            else:
                follower_speed = self.k.vehicle.get_speed(follower)
                # follow_head = self.k.vehicle.get_headway(follower)
                # follower_oriention = self.k.vehicle.get_orientation(follower)[2]
                if follower in self.k.vehicle.get_rl_ids():
                    # follower_oriention = self.k.vehicle.get_orientation(follower)[2]
                    follower_position = self.k.vehicle.get_orientation(follower)
                    follower_x = ego_orientation[0] - follower_position[0]
                    follower_y = ego_orientation[1] - follower_position[1]
                    follower_head  = np.sqrt(pow(follower_x, 2) + pow(follower_y, 2))
                    follower_orientation = follower_position[2]

                else:
                    follower_head = X_LENGTH
                    follower_orientation = rl_orientation

            observation = np.array([
                this_speed / MAX_SPEED,
                (lead_speed - this_speed) / MAX_SPEED,
                lead_head / X_LENGTH,
                (this_speed - follower_speed) / MAX_SPEED,
                follower_head / X_LENGTH,
                rl_orientation / MAX_ANGLE,
                lead_orientation / MAX_ANGLE,
                follower_orientation / MAX_ANGLE
            ])
            for i in range(8):
                observation[i] = min(max(observation[i], -2), 2)
            obs.update({rl_id: observation})
            # print(rl_id, this_speed, lead_id, lead_speed, lead_head, lead_orientation, follower, follower_speed, follower_head, follower_orientation)
            # print(rl_id, observation)
            # time.sleep(1)

        return obs

    # def get_state(self):
    #     """See class definition."""
    #     obs = {}
    
    #     # normalizing constants
    #     max_speed = MAX_SPEED
    #     max_length = X_LENGTH
    #     # X_LENGTH = self.x_length
    #     # Y_LENGTH = self.y_length
    #     # print(self.k.vehicle.get_rl_ids())
    
    #     for vehicle_id in self.k.vehicle.get_rl_ids():
    #         vehicle_state = []
    #         vehicle_speed = self.k.vehicle.get_speed(vehicle_id)
    #         vehicle_oriention = self.k.vehicle.get_orientation(vehicle_id)
    #         ego_state = [vehicle_speed / max_speed, vehicle_oriention[0] / X_LENGTH, vehicle_oriention[1] / Y_LENGTH, vehicle_oriention[2] / MAX_ANGLE]
    #         # vehicle_state += [
    #         #     vehicle_speed / max_speed,
    #         #     vehicle_oriention[0] / X_LENGTH,
    #         #     vehicle_oriention[1] / Y_LENGTH,
    #         #     vehicle_oriention[2] / MAX_ANGLE
    #         # ]
    
    #         # if rl_actions is None:
    
    #         # else:
    #         #     print(rl_actions)
    
    #         for other_id in self.k.vehicle.get_rl_ids():
    #             if vehicle_id != other_id:
    #                 other_speed = self.k.vehicle.get_speed(other_id)
    #                 other_oriention = self.k.vehicle.get_orientation(other_id)
    #                 vehicle_state += [
    #                     (other_speed) / max_speed,
    #                     (vehicle_oriention[0] - other_oriention[0]) / X_LENGTH,
    #                     (vehicle_oriention[1] - other_oriention[1]) / Y_LENGTH,
    #                     (vehicle_oriention[2] - other_oriention[2]) / MAX_ANGLE
    #                 ]
    #             # vehicle_state.update({vehicle_id: [vehicle_speed] + vehicle_oriention})
    #         row = int(len(vehicle_state) / STATE_DIM)
    #         vehicle_state = np.array(vehicle_state)
    #         vehicle_state = vehicle_state.reshape(row, STATE_DIM)
    
    #         for i in range(row):
    #             for j in range(row-i-1):
    #                 if pow(vehicle_state[j][1], 2) + pow(vehicle_state[j][2], 2) > pow(vehicle_state[j+1][1], 2) + pow(vehicle_state[j+1][2], 2):
    #                     tmp = copy.copy(vehicle_state[j])
    #                     vehicle_state[j] = vehicle_state[j+1]
    #                     vehicle_state[j+1] = tmp
    
    #         vehicle_state = vehicle_state.flatten()
    
    #         if len(vehicle_state) < STATE_DIM*NUM_VEHICLE:
    #             vehicle_state = np.concatenate((vehicle_state, np.zeros(STATE_DIM*NUM_VEHICLE-len(vehicle_state))))
    #         else:
    #             vehicle_state = vehicle_state[:STATE_DIM*NUM_VEHICLE]
    
    #         vehicle_state = np.concatenate((vehicle_state, np.array(ego_state)))
    #         obs.update({vehicle_id: vehicle_state})
            
    #         # print(vehicle_id, vehicle_state)
    #         # time.sleep(1)

    #     return obs


    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues action for each traffic light agent.
        """
        # for rl_id, rl_action in rl_actions.items():
        #     i = int(rl_id.split("center")[ID_IDX])
        #     if self.discrete:
        #         raise NotImplementedError
        #     else:
        #         # convert values less than 0.0 to zero and above to 1. 0's
        #         # indicate that we should not switch the direction
        #         action = rl_action > 0.0

        #     if self.currently_yellow[i] == 1:  # currently yellow
        #         self.last_change[i] += self.sim_step
        #         # Check if our timer has exceeded the yellow phase, meaning it
        #         # should switch to red
        #         if self.last_change[i] >= self.min_switch_time:
        #             if self.direction[i] == 0:
        #                 self.k.traffic_light.set_state(
        #                     node_id='center{}'.format(i), state="GrGr")
        #             else:
        #                 self.k.traffic_light.set_state(
        #                     node_id='center{}'.format(i), state='rGrG')
        #             self.currently_yellow[i] = 0
        #     else:
        #         if action:
        #             if self.direction[i] == 0:
        #                 self.k.traffic_light.set_state(
        #                     node_id='center{}'.format(i), state='yryr')
        #             else:
        #                 self.k.traffic_light.set_state(
        #                     node_id='center{}'.format(i), state='ryry')
        #             self.last_change[i] = 0.0
        #             self.direction[i] = not self.direction[i]
        #             self.currently_yellow[i] = 1

        if rl_actions:
            for rl_id, actions in rl_actions.items():
                # self.k.vehicle.apply_speed(rl_id, actions[0]*20)
                # if actions[0] > 0:
                #     accel = actions[0]
                # else:
                #     accel = actions[0]*100
                accel = actions[0]
                # this_speed = self.k.vehicle.get_speed(rl_id)
                self.k.vehicle.apply_acceleration(rl_id, accel*5)
                # print(rl_id, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # if rl_actions is None:
        #     return {}

        # if self.env_params.evaluate:
        #     rew = -rewards.min_delay_unscaled(self)
        # else:
        #     rew = -rewards.min_delay_unscaled(self) \
        #           + rewards.penalize_standstill(self, gain=0.2)

        # # each agent receives reward normalized by number of lights
        # rew /= self.num_traffic_lights

        # rews = {}
        # for rl_id in rl_actions.keys():
        #     rews[rl_id] = rew
        # return rews

        if rl_actions is None:
            return {}
        rewards = {}
        reward_all = []
        target_speed = 5 # self.k.scenario.max_speed()
        # max_length = self.k.scenario.length()
        # speed_mean = np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_rl_ids()))
        for rl_id in self.k.vehicle.get_rl_ids():
            ego_reward = 0
            ego_speed = self.k.vehicle.get_speed(rl_id)
            ego_orientation = self.k.vehicle.get_orientation(rl_id)
            # ego_edge = self.k.vehicle.get_edge(rl_id)
            if ego_speed < 2:
                r_speed = -0.5*((2-ego_speed) / target_speed)
            elif ego_speed > target_speed:
                r_speed = 0.3 - (0.8*(ego_speed - target_speed))/target_speed
            else:
                r_speed = 0.3*pow((ego_speed / target_speed), 3)
            
            # if ego_speed < 2:
            #     r_speed = -1*((2-ego_speed) / target_speed)
            # elif ego_speed > target_speed:
            #     r_speed = 0.1 - (1*(ego_speed - target_speed))/target_speed
            # else:
            #     r_speed = 0.1*(ego_speed / target_speed)


            lead_id = self.k.vehicle.get_leader(rl_id)
            # if lead_id in ["", None]:
            #     lead_speed = MAX_SPEED
            #     lead_head = X_LENGTH
            #     lead_orientation = ego_orientation
            # else:
            #     lead_speed = self.k.vehicle.get_speed(lead_id)
            #     # lead_head = self.k.vehicle.get_headway(rl_id)
            #     # lead_orientation = self.k.vehicle.get_orientation(lead_id)
            #     lead_position = self.k.vehicle.get_orientation(lead_id)
            #     lead_orientation = lead_position[2]
            #     lead_x = ego_orientation[0] - lead_position[0]
            #     lead_y = ego_orientation[1] - lead_position[1]
            #     lead_head  = np.sqrt(pow(lead_x, 2) + pow(lead_y, 2))
            if not lead_id in ["", None]:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_position = self.k.vehicle.get_orientation(lead_id)
                lead_orientation = lead_position[2]
                lead_x = ego_orientation[0] - lead_position[0]
                lead_y = ego_orientation[1] - lead_position[1]
                lead_head = np.sqrt(pow(lead_x, 2) + pow(lead_y, 2))
                TTC = lead_head / (ego_speed - lead_speed + 1e-5)

                if lead_orientation == ego_orientation:
                    # if TTC < 1 and TTC > 0:
                    if ego_speed > lead_speed and TTC < TTC_HEAD:
                        ego_reward = max(-1 / pow(TTC/TTC_HEAD, 2), -100)
                        # ego_reward = max(-1 / TTC, -100)
                        # print(rl_id, lead_id, ego_reward, "1")
                        # time.sleep(3)
                    else:
                        ego_reward = r_speed
                        # print(rl_id, ego_reward, "2")
                else:
                    if lead_head < LEAD_HEAD and lead_head > 0:
                        # print(lead_head, ego_speed)
                        ego_reward = max(-0.2*ego_speed / pow(lead_head/LEAD_HEAD, 2), -100)
                        # ego_reward = max(-ego_speed / lead_head, -100)
                        # print(rl_id, lead_id, ego_reward, "3")
                        # time.sleep(3)
                    else:
                        ego_reward = r_speed
                        # print(rl_id, ego_reward, "4")
            else:
                ego_reward = r_speed
            reward = np.clip(ego_reward, -100, 1)
            # if reward == None:
            #     reward = 0

            rewards[rl_id] = reward
            reward_all.append(reward)
            
            # pbt for policy diversity
            # rewards[rl_id] = 0
            # # rewards_old = rewards.copy()
            # ego_edge = self.k.vehicle.get_edge(rl_id)
            # ego_orientation = self.k.vehicle.get_orientation(rl_id)
            # x_ego = ego_orientation[0]
            # y_ego = ego_orientation[1]
            # # if not rl_id in Flags:
            # if ego_edge in OUTFLOW_EDGE:
            #     if np.abs(x_ego - X_LENGTH/2) > GOAL or np.abs(y_ego - X_LENGTH/2) > GOAL:
            #         rewards[rl_id] = 1
                    # Flags[rl_id] = 1
                        # print(rewards)
                        # time.sleep(1)

        # rewards_old = rewards.copy()
        # print(rewards_old)
        # rl_arrive = self.k.vehicle.get_arrived_rl_ids()
        # if not rl_arrive in ["", None] and rl_arrive != []:
        #     for i in rl_arrive:
        #         rewards[i] = 1
        #         print(rewards)
            # if rewards == rewards_old:
            #     print("True")
            #     print(rewards)
            #     print(rewards_old)
        #     print("rl_arrive", rl_arrive, rewards)
        #     time.sleep(1)
            # print(rl_id, ego_speed, lead_id, reward)
        # time.sleep(1)
        
        # print(rewards)
        # time.sleep(1)

        # reward_sum = sum(reward_all)
        # reward_len = len(reward_all)
        # # print("sum", reward_sum, "len", reward_len)
        # for id in rewards:
        #     # print("#####1#####", rewards[id])
        #     if reward_len > 1:
        #         ratio = ((reward_sum - rewards[id]) / (reward_len - 1)) / rewards[id]
        #         rewards[id] -= 0.04 * abs(math.pi/4 - math.atan(ratio))
                # rewards[id] -= 0.04 * abs(0 - math.atan(ratio))
                # print("ratio", ratio, "reward_i", rewards[id])
            # print("#####2#####", rewards[id])
        return rewards


    def vehicle_speeds(self):
        # speeds = []
        # for rl_id in self.k.vehicle.get_rl_ids():
        #     speeds.append(self.k.vehicle.get_speed(rl_id))
        #     #print(self.k.vehicle.get_speed(rl_id))
        # mean_speed = np.mean(speeds)
        mean_speed = np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_rl_ids()))
        return mean_speed

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        # for veh_ids in self.observed_ids:
        #     for veh_id in veh_ids:
        #         self.k.vehicle.set_observed(veh_id)
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)