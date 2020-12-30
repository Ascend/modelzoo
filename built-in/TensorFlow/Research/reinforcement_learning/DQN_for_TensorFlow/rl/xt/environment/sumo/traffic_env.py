# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI gym environment for SUMO. Run this file for a demo."""

from __future__ import division, print_function, absolute_import

import os
import sys
import datetime
from collections import namedtuple
from math import sqrt

import numpy as np

from xt.environment.sumo.rl_utils import ResultRecorder, RewardRecorder
import xt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    dir = "/home/sumo-master"
    os.environ['SUMO_HOME'] = dir

    dir = "/home/sumo-master/bin:"+os.environ['PATH']
    os.environ['PATH'] = dir
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print("please declare environment variable 'SUMO_HOME'")

try:
    import traci
    import sumolib
except ImportError as error:
    print("Failed to import traci python libs, try setting $SUMO_HOME")

# Default environment configuration
EgoVehicle = namedtuple('EgoVehicle', ['vehID', 'routeID', 'typeID', 'departPos', 'departSpeed', \
                                       'arrivalPos', 'departLane'])
sumo_env_path = os.path.dirname(xt.__file__)
ENV_CONFIG = {
    "sumo_cfg_file":
    os.path.join(sumo_env_path, 'environment/sumo/sumo_env', "sumo.sumocfg"),
    "ego_vehicle_init": [
        EgoVehicle('ego_car', 'left', 'veh_passenger', 170., 0., 10., "1"),
        EgoVehicle('ego_car', 'mid', 'veh_passenger', 170., 0., 10., "0"),
        EgoVehicle('ego_car', 'right', 'veh_passenger', 170., 0., 10., "0")
    ],
    "mode":
    "cli",
    "step_length":
    "0.1",
    "simulation_end":
    31536000,
    "discrete_actions":
    True,
    "squash_action_logits":
    False,
}
AVIL_SPEED = np.array([0.0, 3.0, 6.0, 9.0])


class TrafficEnv():
    """TrafficEnv"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config):
        self.config = ENV_CONFIG
        self.sumo_cfg_file = self.config["sumo_cfg_file"]
        self.ego_vehicles = self.config["ego_vehicle_init"]
        self.ego_veh = self.ego_vehicles[0]
        self.goal = None  # 0:left   1:mid   2:right
        self.ego_index = None
        self.mode = self.config["mode"]
        self.step_length = self.config["step_length"]
        self.simulation_end = self.config["simulation_end"]
        self.discrete_actions = self.config["discrete_actions"]

        sumo_nets = [
            f for f in os.listdir(
                os.path.join(sumo_env_path, 'environment/sumo/sumo_env'))
            if f.endswith(".net.xml")
        ]
        print("[sys path]:", sys.path)
        if len(sumo_nets) != 1:
            sys.exit("Expected exactly one *.net.xml file!")

        self.net = sumolib.net.readNet(
            os.path.join(sumo_env_path, 'environment/sumo/sumo_env', sumo_nets[0]),
            withInternal=True,
        )
        self.subscribe_dic = {"lanePosition": 86, "laneID": 80}

        args = [
            "--configuration-file",
            self.sumo_cfg_file,
            "--step-length",
            self.step_length,
            "--collision.action=warn",
            "--eager-insert=true",
            "--ignore-junction-blocker=1",
            # "--time-to-teleport=0",
            # "--lanechange.duration=1",
            # "--time-to-impatience=0"
        ]

        if self.mode == "gui":
            binary = "sumo-gui"
            # args += ["--start", "--quit-on-end"]
            # args += ["--waiting-time-memory=0.1"]
            # args += ["--no-step-log", "--no-warnings"]
        else:
            binary = "sumo"
            args += ["--no-step-log", "--no-warnings"]
            # args += ["--collision.check-junctions", "true"]

        self.sumo_cmd = [binary] + args
        self.sumo_step = 0
        self.n_step = 0
        self.n_success = 0
        self.n_collision = 0
        self.n_timeout = 0
        self.n_dones = 0
        self.max_step = 450
        # self.dt = traci.simulation.getDeltaT()

        self.ego_veh_collision = False

        self.braking_time = 0.0
        self.sumo_running = False
        self.viewer = None
        self.np_random = None

        # RL parameters

        # RL parameters

        self.obj_feature_dim = 6
        self.obj_max_num = 5
        self.obj_input_dim = self.obj_max_num * self.obj_feature_dim
        self.ego_feature_dim = 1
        self.state_dim = self.ego_feature_dim + self.obj_input_dim
        #self.observation_space = spaces.Box(low=-50, high=50, shape=[self.state_dim])
        self.observation_range = 75.
        self.max_speed = 9.
        self.action_dim = 1
        if self.config["discrete_actions"]:
            self.action_dim = len(AVIL_SPEED)
            #self.action_space = spaces.Discrete(self.action_dim)
        #else:
        #self.action_space = spaces.Box(low=0, high=1, shape=[self.action_dim])

        # RL parameters
        self.state_shape = self.obj_feature_dim
        self.state_num = self.obj_max_num
        self.action_num = self.action_dim
        self.goal_shape = 4
        # RL parameters
        self.start_time = 0

        # logs
        self.np_file = "./reward_succ_ratio.npy"
        self.np_data = np.zeros((1, 4))
        self.episode_reward = RewardRecorder()
        self.episode_result = ResultRecorder()
        self.episode_num = 0
        self.log = True
        self.time_start = datetime.datetime.now()

    def reset(self):
        if not self.sumo_running:
            traci.start(self.sumo_cmd)
            self.sumo_running = True
        else:  # Reset vehicles in simulation
            if self.ego_veh.vehID in traci.vehicle.getIDList():
                traci.vehicle.remove(vehID=self.ego_veh.vehID, reason=2)
            traci.simulation.clearPending()
            if self.sumo_step >= self.simulation_end:
                self.sumo_step = 0
                traci.close()
                traci.start(self.sumo_cmd)
                print("~~~ restart env ~~~")
        # self.sumo_step = 0
        self.n_step = 0
        # self.sumo_deltaT = traci.simulation.getDeltaT() / 1000.  # Simulation timestep in seconds
        for i in traci.vehicle.getIDList():
            if i == self.ego_veh.vehID:
                continue
            traci.vehicle.setColor(i, (255, 255, 0))
        for i in range(100):
            traci.simulationStep()

        self.ego_index = 2
        self.ego_veh = self.ego_vehicles[self.ego_index]

        self.ego_veh_collision = False
        depart_speed = np.random.rand(1)[0] * self.max_speed * 0.0
        traci.vehicle.add(
            vehID=self.ego_veh.vehID,
            routeID=self.ego_veh.routeID,
            typeID=str(self.ego_veh.typeID),
            departPos=str(self.ego_veh.departPos),
            departSpeed=str(depart_speed),
            arrivalPos=str(self.ego_veh.arrivalPos),
            departLane=str(self.ego_veh.departLane),
        )
        traci.vehicle.setSpeedMode(vehID=self.ego_veh.vehID,
                                   sm=6)  # All speed checks are off
        # check if inserted successful
        while self.ego_veh.vehID not in traci.simulation.getDepartedIDList():
            traci.simulationStep()
        state, observation = self.observation()
        traci.vehicle.subscribe(
            self.ego_veh.vehID,
            (self.subscribe_dic["lanePosition"], self.subscribe_dic["laneID"]),
        )
        state = list(state)
        self.start_time = traci.simulation.getCurrentTime()
        self.episode_num += 1
        return np.array(state)

    def close(self):
        if self.sumo_running:
            traci.close(False)
            self.sumo_running = False

    def check_collision(self, observation):
        vertices = []
        obs_len = len(observation)
        for i in range(0, obs_len):
            car_length = traci.vehicle.getLength(observation[i][0])
            car_width = traci.vehicle.getWidth(observation[i][0])
            car_x = observation[i][1]
            car_y = observation[i][2]
            car_yaw = observation[i][3]
            temp_x1 = (car_width / 2 - 0.05) * np.cos(car_yaw * np.pi / 180.0)
            temp_y1 = (car_width / 2 - 0.05) * np.sin(car_yaw * np.pi / 180.0)
            car_fr_vertex = np.array([car_x + temp_x1, car_y - temp_y1])
            car_fl_vertex = np.array([car_x - temp_x1, car_y + temp_y1])
            temp_x2 = (car_length) * np.sin(car_yaw * np.pi / 180.0)
            temp_y2 = (car_length) * np.cos(car_yaw * np.pi / 180.0)
            car_br_vertex = np.array(
                [car_fr_vertex[0] - temp_x2, car_fr_vertex[1] - temp_y2])
            car_bl_vertex = np.array(
                [car_fl_vertex[0] - temp_x2, car_fl_vertex[1] - temp_y2])
            vertices.append(
                np.array([
                    car_bl_vertex, car_fl_vertex, car_fr_vertex, car_br_vertex
                ]))
            if i > 0:
                ego_polygon = vertices[0]
                obj_polygon = vertices[i]
                if self.separating_axis_theorem(ego_polygon, obj_polygon):
                    return True
        return False

    def reached_goal(self):
        route = traci.vehicle.getRoute(self.ego_veh.vehID)
        road_id = traci.vehicle.getRoadID(self.ego_veh.vehID)
        lane_pos = traci.vehicle.getLanePosition(self.ego_veh.vehID)
        if road_id == route[-1] and lane_pos >= self.ego_veh.arrivalPos - 2.0:
            return True
        return False

    # TODO: Refine reward function!!
    def get_reward(self):

        # driving_distance = traci.vehicle.getDistance(self.ego_veh.vehID)
        # if self.ego_index == 0:
        #     max_distance = 37
        # elif self.ego_index == 1:
        #     max_distance = 34.5
        # elif self.ego_index == 2:
        #     max_distance = 29

        # reward_list[self.ego_index] = 5 / (max_distance - driving_distance)

        reward = -0.15
        # reward = 0
        if self.ego_veh_collision:
            reward += -650
        elif self.reached_goal():
            reward += 80
        elif self.n_step >= self.max_step * 0.5:
            reward += -0.15
        if self.n_step == self.max_step - 1:
            reward += -650
        # print("leader: ", traci.vehicle.getLeader(self.ego_veh.vehID))
        return reward

    def step(self, action):
        # if traffic light is red and distance from ego_veh to traffic light < 1, ego_veh will stop.
        next_traffic_light = traci.vehicle.getNextTLS(self.ego_veh.vehID)
        if len(next_traffic_light) > 0 and self.ego_index != 2:
            next_traffic_light = next_traffic_light[0]
            while (next_traffic_light[3] != "g" and
                   next_traffic_light[3] != "G") and next_traffic_light[2] <= 9.5:
                # print("while 1", next_traffic_light, traci.vehicle.getSpeed(self.ego_veh.vehID))
                traci.vehicle.slowDown(self.ego_veh.vehID, 0.0, 0.1)
                traci.simulationStep()
                next_traffic_light = traci.vehicle.getNextTLS(self.ego_veh.vehID)
                self.start_time = traci.simulation.getCurrentTime()
                if len(next_traffic_light) > 0:
                    next_traffic_light = next_traffic_light[0]
                else:
                    break

        if self.ego_index == 2:  # turn right when traffic light is red
            next_traffic_light = traci.vehicle.getNextTLS(self.ego_veh.vehID)
            if len(next_traffic_light) > 0:
                next_traffic_light = next_traffic_light[0]
                # print(next_traffic_light)
                while next_traffic_light[3] == "r" and next_traffic_light[2] <= 9.5:
                    # print("while 1", next_traffic_light, traci.vehicle.getSpeed(self.ego_veh.vehID))
                    traci.vehicle.slowDown(self.ego_veh.vehID, 0.0, 0.1)
                    traci.simulationStep()
                    next_traffic_light = traci.vehicle.getNextTLS(
                        self.ego_veh.vehID)
                    self.start_time = traci.simulation.getCurrentTime()
                    if len(next_traffic_light) > 0:
                        next_traffic_light = next_traffic_light[0]
                    else:
                        break

        if not self.sumo_running:
            self.reset()
        self.sumo_step += 1
        self.n_step += 1
        # print("action:", action)
        if action != -1:
            # acceleration = np.clip(action[0], -4.5, 2.6)
            # new_speed = traci.vehicle.getSpeed(self.ego_veh.vehID) + traci.simulation.getDeltaT() * acceleration
            if self.config["discrete_actions"]:
                new_speed = AVIL_SPEED[action]
            else:
                new_speed = np.clip(action[0] * self.max_speed, 0,
                                    self.max_speed)
            # ego_speed = traci.vehicle.getSpeed(self.ego_veh.vehID)
            # print("action", new_speed, " speed:", ego_speed, "m/s")
            # TODO check if how the speed works
            # traci.vehicle.setSpeed(self.ego_veh.vehID, new_speed)
            new_speed = float(new_speed)
            traci.vehicle.slowDown(self.ego_veh.vehID, new_speed, 0.1)

        traci.simulationStep()
        state, observation = self.observation()
        self.ego_veh_collision = self.check_collision(observation)
        reward = self.get_reward()

        # print self.check_collision()
        if self.ego_veh_collision:
            self.n_collision += 1
            print("!!! Collision !!!")
        if self.reached_goal():
            self.n_success += 1
            print("~~~ Well done ~~~")
        if self.n_step >= self.max_step:
            self.n_timeout += 1
            print("!!! Time  out !!!")
        done = (self.ego_veh_collision or self.reached_goal()
                or (self.sumo_step > self.simulation_end)
                or (self.n_step >= self.max_step))
        intersection_distance = -1
        if (self.ego_veh.vehID in traci.vehicle.getIDList()):
            subscription_result = traci.vehicle.getSubscriptionResults(
                self.ego_veh.vehID)
            intersection_distance = (
                self.net.getEdge(subscription_result[
                    self.subscribe_dic["laneID"]]).getLength() -
                subscription_result[self.subscribe_dic["lanePosition"]])

        if self.log:
            self.episode_reward.add_rewards(reward)
            if done:
                self.episode_reward.start_new_episode()
                result = 1
                if self.reached_goal():
                    result = 2
                self.episode_result.add_result(result, self.ego_index)
                self.np_data = np.r_[self.np_data, [[
                    self.episode_num, self.episode_reward.mean,
                    self.episode_result.mean(self.ego_index),
                    (datetime.datetime.now() - self.time_start).seconds / 60
                ]]]
                if self.episode_num % 50 == 0:

                    print("Episode: ", self.episode_num, "  Mean: ",
                          self.episode_reward.mean, "  Succ: ",
                          self.episode_result.mean(self.ego_index),
                          "   Time: ",
                          (datetime.datetime.now() - self.time_start).seconds /
                          60)
                    #np.save(self.np_file, self.np_data)
        return np.array(state), reward, done, {}

    @staticmethod
    # convert ENU to FRU-vehicle body coordination
    def enu_to_fru(ego_x, ego_y, ego_yaw, obj_x, obj_y, obj_yaw):
        theta = 2 * np.pi - ego_yaw * np.pi / 180.0
        # print("theta:", theta)
        x_con = (obj_x - ego_x) * (np.cos(theta)) + (obj_y -
                                                     ego_y) * (np.sin(theta))
        y_con = -(obj_x - ego_x) * (np.sin(theta)) + (obj_y -
                                                      ego_y) * (np.cos(theta))
        yaw_con = obj_yaw - ego_yaw

        if yaw_con < -180.0:
            yaw_con += 360
        elif yaw_con > 180.0:
            yaw_con -= 360
        # print(yaw_con)
        return x_con, y_con, np.deg2rad(yaw_con)

    @staticmethod
    def state_process(ego_x, ego_y, ego_yaw, obj_x, obj_y, obj_yaw):
        yaw_con = obj_yaw - ego_yaw
        x_con = obj_x - ego_x
        y_con = obj_y - ego_y
        if yaw_con <= -180.0:
            yaw_con += 360
        elif yaw_con >= 180.0:
            yaw_con -= 360
        # print(yaw_con)
        return x_con, y_con, np.deg2rad(yaw_con)

    def neighbor_filter(self, observation, ego_pos, ego_ang):
        filted_nb = []
        state = []
        num = 0
        for neighbor in observation:
            if num >= self.obj_max_num:
                break
            obj_x, obj_y, obj_yaw = self.enu_to_fru(ego_pos[0], ego_pos[1],
                                                    ego_ang, neighbor[1],
                                                    neighbor[2], neighbor[3])

            is_discard = (obj_yaw <= 0.1 and abs(obj_x) >= 1)
            is_retrograde = (obj_yaw >= (np.pi * 1 / 2 + 0.07))
            is_retrograde = is_retrograde and (obj_yaw < np.pi)
            is_discard = is_discard or is_retrograde
            is_discard = is_discard or (obj_y < -5.0)
            if (is_discard):
                traci.vehicle.setColor(neighbor[0], (255, 0, 0))
                # print("discard:   ", neighbor[0], obj_x, obj_y, obj_yaw, neighbor[3], ego_ang)
            else:
                state.extend([
                    obj_x, obj_y, neighbor[4],
                    np.sin(obj_yaw),
                    np.cos(obj_yaw)
                ])
                traci.vehicle.setColor(neighbor[0], (255, 0, 255))
                # print("hold:   ", neighbor[0], obj_x, obj_y, obj_yaw)
                num += 1
        for i in range(self.obj_max_num - num):
            state.extend([
                -self.observation_range, -self.observation_range, 0, 0.0, -1.0,
                0.0
            ])
        return state

    def back_samedirection_filter(self, observation, ego_pos, ego_ang):
        state = []
        num = 0
        for neighbor in observation:
            if num >= self.obj_max_num:
                break
            obj_x, obj_y, obj_yaw = self.enu_to_fru(ego_pos[0], ego_pos[1],
                                                    ego_ang, neighbor[1],
                                                    neighbor[2], neighbor[3])
            if (obj_y < -5.0) or (abs(obj_yaw) <= 0.1 and
                                  abs(obj_x) >= 1):  # back and same direction
                traci.vehicle.setColor(neighbor[0], (255, 0, 0))
            else:
                if obj_x < -2:
                    direction_flag = 0  # left of the ego
                elif obj_x > 2:
                    direction_flag = 2  # right of the ego
                else:
                    direction_flag = 1
                state.extend([
                    obj_x,
                    obj_y,
                    direction_flag,
                    neighbor[4],
                    np.sin(obj_yaw),
                    np.cos(obj_yaw),
                ])
                traci.vehicle.setColor(neighbor[0], (255, 0, 255))
                num += 1
        for i in range(self.obj_max_num - num):
            state.extend([
                -self.observation_range, -self.observation_range, 0, 0.0, -1.0,
                0.0
            ])
        return state

    def observation(self):
        state = []
        # observation = []
        visible = []
        ego_car_in_scene = False
        if self.ego_veh.vehID not in traci.vehicle.getIDList():
            self.reset(self.goal)
        ego_pos = traci.vehicle.getPosition(self.ego_veh.vehID)
        ego_ang = traci.vehicle.getAngle(self.ego_veh.vehID)
        ego_speed = traci.vehicle.getSpeed(self.ego_veh.vehID)
        state.extend([ego_speed / self.max_speed])
        ego_car_in_scene = True

        for i in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(i)
            pos = traci.vehicle.getPosition(i)
            angle = traci.vehicle.getAngle(i)
            # laneid = traci.vehicle.getRouteID(i)
            observation_tuple = (i, pos[0], pos[1], angle, speed)
            # observation.append(observation_tuple)
            if ego_car_in_scene and i not in self.ego_veh.vehID:
                traci.vehicle.setColor(i, (255, 255, 0))
                dist = np.linalg.norm(np.asarray(pos) - np.asarray(ego_pos))
                if dist < self.observation_range:  # 75 is 75 meters
                    observation_tuple += (dist, )
                    visible.append(observation_tuple)
                    traci.vehicle.setColor(i, (0, 255, 0))
                    # observation.append(observation_tuple)

        observation = sorted(visible, key=lambda x: x[:][-1], reverse=False)
        state += self.back_samedirection_filter(observation, ego_pos, ego_ang)
        observation.insert(
            0,
            (self.ego_veh.vehID, ego_pos[0], ego_pos[1], ego_ang, ego_speed))
        state = np.reshape(state, self.state_dim)

        return state, observation

    ##############################
    ### separation_axis_theorem###
    ##############################
    def normalize(self, value):
        norm = sqrt(value[0]**2 + value[1]**2)
        return (value[0] / (norm + 0.001), value[1] / (norm + 0.001))

    def dot(self, point_a, point_b):
        return point_a[0] * point_b[0] + point_a[1] * point_b[1]

    def edge_direction(self, point_0, point_1):
        return (point_1[0] - point_0[0], point_1[1] - point_0[1])

    def orthogonal(self, value):
        return (value[1], -value[0])

    def vertices_to_edges(self, vertices):
        return [
            self.edge_direction(vertices[i], vertices[(i + 1) % len(vertices)])
            for i in range(len(vertices))
        ]

    def project(self, vertices, axis):
        dots = [self.dot(vertex, axis) for vertex in vertices]
        return [min(dots), max(dots)]

    def contains(self, number, range_):
        num_a = range_[0]
        num_b = range_[1]
        if num_b < num_a:
            num_a = range_[1]
            num_b = range_[0]
        return (number >= num_a) and (number <= num_b)

    def overlap(self, point_a, point_b):
        if self.contains(point_a[0], point_b):
            return True
        if self.contains(point_a[1], point_b):
            return True
        if self.contains(point_b[0], point_a):
            return True
        if self.contains(point_b[1], point_a):
            return True
        return False

    def separating_axis_theorem(self, vertices_a, vertices_b):
        # print("collistion a:", vertices_a, "    b:", vertices_b)
        edges_a = self.vertices_to_edges(vertices_a)
        edges_b = self.vertices_to_edges(vertices_b)
        edges = edges_a + edges_b
        axes = [self.normalize(self.orthogonal(edge)) for edge in edges]
        len_axes = len(axes)
        for i in range(len_axes):
            projection_a = self.project(vertices_a, axes[i])
            projection_b = self.project(vertices_b, axes[i])
            overlapping = self.overlap(projection_a, projection_b)
            if not overlapping:
                return False
        return True
