#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# ==============================================================================
#
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
Utils for profiling status
"""
from collections import deque
from time import time

import numpy as np


class LoopTracker(object):
    """timekeeping, contains
    1) with `enter`-> `exit`; 2) loop between current and next `exit`. """

    def __init__(self, length):
        self.with_time_list = deque(maxlen=length)
        self.loop_time_list = deque(maxlen=length)
        self.loop_point = None
        
    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        self.with_time_list.append(self.end - self.start)

        if not self.loop_point:
            self.loop_point = self.end
        else:
            self.loop_time_list.append(self.end - self.loop_point)
            self.loop_point = self.end

    def average(self, time_name):
        """mean time of `with` interaction, and loop time as well."""
        if time_name == "enter":
            return np.nanmean(self.with_time_list) * 1000
        elif time_name == "loop":
            return np.nanmean(self.loop_time_list) * 1000
        else:
            return np.nan


class SingleTracker(object):
    """single time tracker, only profiling the enter time used."""

    def __init__(self, length):
        self.with_time_list = deque(maxlen=length)
        self.start = time()

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.with_time_list.append(time() - self.start)

    def average(self):
        """mean time of `with` interaction"""
        if not self.with_time_list:
            return np.nan
        return np.nanmean(self.with_time_list) * 1000


class PredictStats(object):
    """predictor status records
    handle the wait and inference time of predictor"""

    def __init__(self):
        """init with default value"""
        self.obs_wait_time = 0.0
        self.inference_time = 0.0
        self.iters = 0.0

    def get(self):
        ret = {
            "mean_predictor_wait_ms": self.obs_wait_time * 1000 / self.iters,
            "mean_predictor_infer_ms": self.inference_time * 1000 / self.iters,
        }
        self.reset()
        return ret

    def reset(self):
        self.obs_wait_time = 0.0
        self.inference_time = 0.0
        self.iters = 0.0


class AgentStats(object):
    """ Agent status records
    handle the env.step and inference time of Agent"""

    def __init__(self):
        """init with default value"""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0.0

    def get(self):
        """get agent status and clear the buffer"""
        ret = {
            "mean_env_step_time_ms": self.env_step_time * 1000 / self.iters,
            "mean_inference_time_ms": self.inference_time * 1000 / self.iters,
            "iters": self.iters,
        }

        self.reset()
        return ret

    def reset(self):
        """reset buffer"""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0


class AgentGroupStats(object):
    """ AgentGroup status records
    handle the env.step and inference time of AgentGroup
    the status could been make sence within once explore
    There should been gather by logger or others"""

    def __init__(self, n_agents, env_type):
        """init with default value"""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0
        self.explore_time_in_epi = 0.0
        self.wait_model_time = 0.0
        self.restore_model_time = 0.0

        self.n_agents = n_agents
        self.env_api_type = env_type
        self._stats = dict()

    def update_with_agent_stats(self, agent_stats: list):
        """update agent status to agent group"""
        _steps = [sta["mean_env_step_time_ms"] for sta in agent_stats]
        _infers = [sta["mean_inference_time_ms"] for sta in agent_stats]
        _iters = [sta["iters"] for sta in agent_stats]
        self._stats.update(
            {
                "mean_env_step_ms": np.nanmean(_steps),
                "mean_inference_ms": np.nanmean(_infers),
                "iters": np.max(_iters),
            }
        )

    def get(self):
        """get the newest one-explore-status of agent group"""
        self._stats.update(
            {
                "explore_ms": self.explore_time_in_epi * 1000,
                "wait_model_ms": self.wait_model_time * 1000,
                "restore_model_ms": self.restore_model_time * 1000,
            }
        )
        # use unified api, agent group will record the iteraction times.
        if self.iters > 1e-1:
            self._stats.update(
                {
                    "mean_env_step_time_ms": self.env_step_time * 1000 / self.iters,
                    "mean_inference_time_ms": self.inference_time * 1000 / self.iters,
                    "iters": self.iters,
                }
            )

        self.reset()
        return self._stats

    def reset(self):
        """reset buffer."""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0
        self.explore_time_in_epi = 0.0
        self.wait_model_time = 0.0
        self.restore_model_time = 0.0
