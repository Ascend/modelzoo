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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test run one episode
"""

import tensorflow as tf
import numpy as np

from xt.algorithm.qmix.qmix import QMixAlgorithm, QMixAgent
from tests.qmix.set_2s_vs_1sc_paras import get_args, get_scheme
import unittest
from xt.algorithm.qmix.transforms import OneHotNp
import os

NUM_PARALLEL_EXEC_UNITS = 4
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
# os.environ["KMP_AFFINITY"] = "verbose" # no affinity
# os.environ["KMP_AFFINITY"] = "none" # no affinity
os.environ["KMP_AFFINITY"] = "disabled"  # completely disable thread pools


class TestQMix(unittest.TestCase):
    def setUp(self) -> None:

        self.obs_shape = 26
        self.avail_action_num = 7
        self.args = get_args()
        # mac use buffer.scheme, runner used scheme.
        # buffer.scheme
        self.scheme = get_scheme()
        # self.qmix = QMixAlgorithm(
        #     self.scheme, self.args, self.avail_action_num, tf.float32
        # )
        self.agent = QMixAgent(args=self.args, scheme=self.scheme)

    # @unittest.skip("skip passed")
    def test_run_one_episode(self):
        seq_max = 1
        args = self.args
        env_info = self.agent.env.get_env_info()
        groups = {"agents": args.n_agents}

        preprocess_np = {
            "actions": ("actions_onehot", [OneHotNp(out_dim=args.n_actions)])
        }
        scheme_np = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": np.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": np.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": np.uint8},
        }

        self.agent.setup(scheme=scheme_np, groups=groups, preprocess=preprocess_np)

        batch = self.agent.run_one_episode(test_mode=False)
        print(batch)
        # for k in (batch.scheme.keys()):
        #     print(k, "*"*20)
        #     print(batch[k])
