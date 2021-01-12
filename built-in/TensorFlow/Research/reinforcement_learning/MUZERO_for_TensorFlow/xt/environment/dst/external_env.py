# import campus_ros_basic
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
"""
Env module supply one standard interface to the other modules,if
you wants to add your own  environment, you only need to supply interface:
reset, step and get_init_state
"""
from threading import Thread

from xt.environment.dst.policy_server import PolicyServer
from xt.environment.environment import Environment
from xt.framework.comm.uni_comm import UniComm
from xt.framework.register import Registers


@Registers.env.register
class ExternalEnv(Environment):
    """
    external env class
    """
    def __init__(self, env_info, **kwargs):
        """
        :param env_info: the config info of environment
        """
        self.env_info = env_info
        self.episode_id = 0
        env_info['port'] += (env_info['agent_num']) * kwargs['env_id']
        super(ExternalEnv, self).__init__(env_info, **kwargs)

    def init_env(self, env_info):
        """
        :param env_name:
        :param env_info:
        """
        self.port = env_info['port']
        self.env_server = UniComm('CommByZmq', type='REP', port=self.port + 1)
        self.policy_server = PolicyServer(address="127.0.0.1", port=self.port)

        policy_thread = Thread(target=self.policy_server.serve_forever)
        policy_thread.start()

    def reset(self):
        """
        reset
        """
        pass

    def step(self, action, agent_index=0):
        """
        step
        """
        self.env_server.send(action)
        client_dict = self.env_server.recv()
        command = client_dict['command']
        if command == 'GET_ACTION':
            obs = client_dict["observation"]
            state = self.transfer_state(obs)
            reward = client_dict["reward"]
            info = client_dict["info"]
            done = False
        elif command == 'END_EPISODE':
            print("get reboot client")
            self.reboot = True
            obs = client_dict["observation"]
            state = self.transfer_state(obs)
            reward = 0
            done = True
            info = {}
            self.env_server.send(self.episode_id)
        else:
            raise ValueError("get wrong command ", client_dict['command'])

        return state, reward, done, info

    def close(self):
        print("shutdown XingTian server, port: ",
              self.port, "-", self.port + self.env_info.get('agent_num', 1) - 1)

    def reset_env(self, agent_id):
        client_dict = self.env_server.recv()
        command = client_dict['command']
        if command != "START_EPISODE":
            raise Exception("Unknown command: {}".format(command))
        self.env_server.send(self.id)
        self.episode_id += 1

        client_dict = self.env_server.recv()
        command = client_dict['command']
        if command != "GET_ACTION":
            raise Exception("Unknown command: {}".format(command))
        state = self.transfer_state(client_dict["observation"])
        self.init_state = state

        return self.init_state

    @staticmethod
    def transfer_state(state, *args):
        """
        Override this to implement the state transform
        """
        return state
