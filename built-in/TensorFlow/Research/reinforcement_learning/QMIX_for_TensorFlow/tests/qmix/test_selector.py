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
test action selector with random
"""

import os
import unittest

import numpy as np

from tests.qmix.set_2s_vs_1sc_paras import get_args, get_scheme
from xt.algorithm.qmix.qmix import EpsilonGreedyActionSelector


class TestQMix(unittest.TestCase):
    def setUp(self) -> None:
        self.args = get_args()
        self.selector = EpsilonGreedyActionSelector(self.args)

    def test_corner_select(self):
        out_val = np.array(
            [
                [
                    [
                        0.28811422,
                        0.17851579,
                        0.22732314,
                        0.6934695,
                        -0.2829191,
                        -0.21144678,
                        -0.447422,
                    ],
                    [
                        -0.12816036,
                        0.05661841,
                        0.01789608,
                        0.16747335,
                        0.11160274,
                        -0.11278875,
                        -0.04281046,
                    ],
                ]
            ]
        )

        t_env = 424

        avail_actions = np.array([[[0, 1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0]]])

        for _ in range(100):
            select_actions = self.selector.select_action(
                out_val, avail_actions, t_env, test_mode=False
            )
            g = self._check_in_int(select_actions)
            print(g)

    @staticmethod
    def _check_in_int(action):
        gap = 0
        for i in range(2):
            gap += np.abs(int(action[0][i]) - action[0][i])
        if gap > 1e-5:
            raise ValueError("action: {} with gap: {}".format(action, gap))
        return gap


if __name__ == "__main__":
    unittest.main()
