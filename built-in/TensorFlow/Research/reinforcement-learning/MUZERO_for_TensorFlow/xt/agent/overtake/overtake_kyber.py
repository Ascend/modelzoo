#!/usr/bin/env python
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
""" OverTake agent for kyber sim."""

import random
import numpy as np

from xt.agent import Agent
from xt.framework.register import Registers
from xt.framework.comm.message import message

VIEW_RANGE = 125.0  # meter
TIME_RANGE = 12.5  # seconds
SPEED_RANGE = 300.0


@Registers.agent.register
class OverTakeKyber(Agent):
    """
    OverTakeKyber class base on AsyncAgent class
    """
    def __init__(self, env, alg, agent_config, **kwargs):
        super(OverTakeKyber, self).__init__(env, alg, agent_config, **kwargs)
        self.epsilon = 1.0
        self.episode_count = agent_config.get("episode_count", 100000)
        self.goal_lane = 1
        self.last_lane_remains = 2500.0
        self.step = 0
        self.track_length = 6000.0
        self.lanes_num = 3.0
        self.total_change_steps = 0
        self.total_change_count = 0
        self.ave_steps = 0
        self.test_info = ["str", "turn", "ab", "ab_1", "rel", "step", "real_step"]

    def infer_action(self, state, use_explore):
        s_t = self._state_convert(state)
        # if explore action
        if use_explore and random.random() < self.epsilon:
            action = np.random.randint(0, 3)
        elif use_explore:
            # Get Q values with deliver for each action.
            send_data = message(s_t, agent_id=self.id)
            self.send_explorer.send(send_data)
            action = self.recv_explorer.recv()
        else:
            action = self.alg.predict(s_t)

        # update episode value
        if use_explore:
            self.epsilon -= 1.0 / self.episode_count
            self.epsilon = max(0.01, self.epsilon)

        # update transition data
        self.transition_data.update({
            "cur_state": s_t,
            "action": action,
        })
        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        s_t1 = self._state_convert(next_raw_state)
        if use_explore:
            r_t = self._get_reward(
                self.transition_data["cur_state"],
                s_t1,
                self.transition_data["action"],
                info,
            )
        else:
            r_t = self._get_test_reward(next_raw_state)

        self.transition_data.update({"next_state": s_t1, "reward": r_t, "done": done, "info": info})

        # deliver this transition data to learner, trigger train process.
        if use_explore:
            train_data = {k: [v] for k, v in self.transition_data.items()}
            train_data = message(train_data, agent_id=self.id)
            self.send_explorer.send(train_data)

        return self.transition_data

    def do_one_interaction(self, raw_state, use_explore=True):
        """Use the Agent do once interaction.
        Owing to the env.step is non-normal,
        here, rewrite the `do_one_interaction` function.
        :param raw_state:
        :param use_explore:
        :return:
        """
        action = self.infer_action(raw_state, use_explore)

        next_raw_state, reward, done, info = self._agent_step(raw_state, action)

        # next_raw_state, reward, done, info = self.env.step(action)
        return self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)

    @staticmethod
    def _get_reward(state, new_state, action, info):
        """refer to AgentBase"""
        # state = self.convert(state)
        # new_state = self.convert(new_state)
        speed_cur = state[6] * SPEED_RANGE
        speed_sec = speed_cur / 3.6
        front_car_left = state[0] * TIME_RANGE * speed_sec
        front_car_mid = state[1] * TIME_RANGE * speed_sec + 0.1
        front_car_right = state[2] * TIME_RANGE * speed_sec
        speed = new_state[6]
        speed_ = speed * 300.0

        if info[-1][1] is False:
            reward = -100
        elif action != 0:
            k_l = (np.clip((front_car_left / front_car_mid), 0.5, 1.5) if action == 1 else np.clip(
                (front_car_right / front_car_mid), 0.5, 1.5))
            if k_l > 1.0:
                k_l = 1.5
            elif k_l < 1.0:
                k_l = 0.5
            reward = 0.3 * k_l * speed_
            if k_l == 1 and state[1] == 0.8:
                reward = -30
        else:
            reward = speed_
        return reward

    @staticmethod
    def _get_test_reward(new_raw_state):
        """refer to AgentBase"""
        sp_ = new_raw_state[3]
        reward = sp_
        return reward

    def _state_convert(self, raw_state):
        """convert raw state to training state"""
        variables_dict = dict()
        variables_dict["s_t"] = np.hstack((0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0, 0.0, 1.0))
        variables_dict["v_t"] = np.hstack((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        variables_dict["add_s_t"] = np.hstack((0.8, 0.8))
        variables_dict["add_v_t"] = np.hstack((0.0, 0.0))
        variables_dict["flag_t"] = 0.0
        variables_dict["add_dist_min"] = np.hstack((1000.0, 1000.0, 1000.0, 1000.0))
        variables_dict["dist_min"] = np.hstack((1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0))
        variables_dict["ego_lane"] = raw_state[16]
        variables_dict["lane_ids"] = raw_state[18]
        variables_dict["ego_lane"] = variables_dict["lane_ids"].index(variables_dict["ego_lane"])
        if variables_dict["ego_lane"] == 0 or variables_dict["ego_lane"] == 2:
            variables_dict["s_t"][variables_dict["ego_lane"]] = 1.0
            variables_dict["s_t"][variables_dict["ego_lane"] + 3] = 1.0
            variables_dict["flag_t"] = 1 if variables_dict["ego_lane"] == 0 else -1

        variables_dict["ego_raw_speed"] = raw_state[3]
        variables_dict["filter_speed"] = (variables_dict["ego_raw_speed"]
                                          if variables_dict["ego_raw_speed"] >= 10.0 else 10.0)
        variables_dict["s_t"][6] = variables_dict["ego_raw_speed"] / SPEED_RANGE
        objects = raw_state[-1]
        # print("ego_speed",ego_raw_speed,"ego_lane",ego_lane)
        if objects[0] is not None:
            # for i in range(len(objects)):
            for i, _object in enumerate(objects):
                lane_id = objects[i][0]
                dist = abs(objects[i][1]) * np.sign(objects[i][1])
                speed = objects[i][2]
                pre_post = np.sign(dist)
                flag = 0 if pre_post == 1.0 else 1

                if abs(dist) < VIEW_RANGE:
                    for j in range(3):
                        adjacent_lane = variables_dict["ego_lane"] - 1 + j
                        dist_index = j + flag * 3
                        if (lane_id == adjacent_lane and abs(dist) < variables_dict["dist_min"][dist_index]):
                            self.min_dist(
                                variables_dict["v_t"],
                                variables_dict["s_t"],
                                dist_index,
                                speed,
                                dist,
                                variables_dict["filter_speed"],
                            )
                            variables_dict["dist_min"][dist_index] = abs(dist)

                    if abs(dist) < variables_dict["add_dist_min"][flag]:
                        if (variables_dict["ego_lane"] == 0 and lane_id == variables_dict["ego_lane"] + 2
                                or variables_dict["ego_lane"] == len(variables_dict["lane_ids"]) - 1
                                and lane_id == variables_dict["ego_lane"] - 2):
                            self.min_dist(
                                variables_dict["add_v_t"],
                                variables_dict["add_s_t"],
                                flag,
                                speed,
                                dist,
                                variables_dict["filter_speed"],
                            )

        state = np.hstack((
            variables_dict["s_t"],
            variables_dict["v_t"],
            variables_dict["add_s_t"],
            variables_dict["add_v_t"],
            variables_dict["flag_t"],
        ))
        return state

    @staticmethod
    def min_dist(v_t, s_t, flag, speed, dist, f_speed):
        """compute minimum distance"""
        v_t[flag] = np.clip(speed / f_speed, 0.0, 2.0)
        s_t[flag] = np.clip(3.6 * abs(dist) / (f_speed * TIME_RANGE), 0, 0.8)

    @staticmethod
    def collision_detection(objects, target_lane):
        """detect collision"""
        if objects is list:
            return False
        value = False
        goal_front_dist = 1000
        goal_back_dist = -1000
        # for index in range(len(objects)):
        for index, _object in enumerate(objects):
            lane_id = objects[index][0] + 1
            dist = abs(objects[index][1]) * np.sign(objects[index][1])
            if lane_id == target_lane and 0 < dist < goal_front_dist:
                goal_front_dist = dist
            if lane_id == target_lane and 0 > dist > goal_back_dist:
                goal_back_dist = dist

        if goal_front_dist < 6 or goal_back_dist > -6:
            value = True
        return value

    def check_state(self, state, action):
        """check_state"""
        if action == 0:
            return "str"
        s_t = self._state_convert(state)
        front_car_left = s_t[0]
        front_car_mid = s_t[1]
        front_car_right = s_t[2]

        if front_car_mid == 0.8:
            return "ab"
        if (action == 1 and front_car_left == 1.0) or (action == 2 and front_car_right == 1.0):
            return "ab"
        if (action == 1 and front_car_left < front_car_mid) or (action == 2 and front_car_right < front_car_mid):
            if state[16] == 1:
                return "ab_1"
            return "rel"

        return "turn"

    def _action_check(self, raw_state, action):
        """check action"""
        variables_dict = dict()
        variables_dict["state_follow"] = 4
        # state_follow = 4
        variables_dict["state_failed"] = 3
        variables_dict["state_process"] = 2
        sc_ = False
        variables_dict["target_lane"] = None
        danger = False
        step_count = 0
        if action == 0:
            variables_dict["env_action"] = 0
            variables_dict["steps"] = random.randint(10, 20)
        else:
            if action == 1:
                variables_dict["env_action"] = 1
                variables_dict["target_lane"] = raw_state[15]
            else:
                variables_dict["env_action"] = 2
                variables_dict["target_lane"] = raw_state[17]
            variables_dict["steps"] = 120
        if action == 0:
            sc_ = True
            for i in range(variables_dict["steps"]):
                state, reward, done, info = self.env.step(variables_dict["env_action"], 0)
                step_count += 1
                if done:
                    break
        else:
            variables_dict["exec_action"] = variables_dict["env_action"]
            for i in range(variables_dict["steps"]):
                state, reward, done, info = self.env.step(variables_dict["exec_action"], 0)
                step_count += 1
                lane_chang_state = state[20]
                cur_lane = state[16]
                if (lane_chang_state == variables_dict["state_failed"] or variables_dict["target_lane"] == -1 or done):
                    break
                if cur_lane == variables_dict["target_lane"]:
                    if lane_chang_state == variables_dict["state_follow"]:
                        sc_ = True
                        break
                    if lane_chang_state == variables_dict["state_process"]:
                        variables_dict["exec_action"] = 0
                objects = state[-1]
                variables_dict["collision"] = self.collision_detection(objects, variables_dict["target_lane"])
                if variables_dict["collision"] is True:
                    danger = True
        return sc_, danger, state, reward, done, info, step_count

    def _agent_step(self, raw_state, action):
        """refer to AgentBase"""
        sc_, danger, state, reward, done, info, step_count = self._action_check(raw_state, action)
        if danger is True:
            sc_ = False

        if state[19][2] < 20:
            done = True
        info = []
        action_info = self.check_state(raw_state, action)
        info.append((action_info, 1))
        info.append(("step", step_count))
        real_count = state[1] - raw_state[1]
        if real_count < 0:
            real_count = state[1]
        info.append(("real_step", real_count))
        info.append(("danger", danger))
        info.append(("sc", sc_))
        if state[1] < raw_state[1]:
            print("#########error count $$$$$$$$$$", state[1], raw_state[1])
        return state, reward, done, info
