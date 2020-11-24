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
"""simulation with torcs"""
from __future__ import division, print_function

from Env_Platform import Env_Platform
from xt.environment.environment import Environment
from xt.framework.register import Registers


@Registers.env
class RlEnvSimu(Environment):
    """ simulator platform for noah case"""

    def init_env(self, env_info):
        """
        create a environment instance

        :param: the config information of environment
        :return: the instance of environment
        """
        env_name = env_info["name"]
        vision = env_info["vision"]
        config = env_info["config"]
        env = Env_Platform(env_name, vision, config)

        self.init_state = None
        return env

    def reset(self, reset_arg=None):
        """
        reset the environment.

        :param reset_arg: optional parameter, it's used to specify the scene,
                such as vehicle starting pose
        :return: the observation of gym environment
        """
        if reset_arg is None:
            state = self.env.reset()
        else:
            state = self.env.reset(reset_arg)
        self.init_state = state
        return state
