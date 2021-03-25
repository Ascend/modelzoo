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
test agent with explore
"""

import tensorflow as tf
import numpy as np

from xt.algorithm.qmix.qmix import QMixAlgorithm
from tests.qmix.set_2s_vs_1sc_paras import get_args, get_scheme
import unittest

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
        self.qmix = QMixAlgorithm(
            self.scheme, self.args, self.avail_action_num, 300, tf.float32
        )

    # @unittest.skip("skip passed")
    def test_rnn_agent_infer(self):
        self.qmix.build_actor_graph()
        with self.qmix.graph.as_default():
            self.qmix.sess.run(tf.global_variables_initializer())

            self.qmix.hi_out_val_default = self.qmix.sess.run(
                self.qmix.gru_cell.zero_state(self.args.n_agents, dtype=tf.float32)
            )
        self.qmix.reset_hidden_state()

        # init data with up
        obs = np.random.rand(1, 1, self.args.n_agents, self.obs_shape)

        with self.qmix.graph.as_default():
            for i in range(3):
                out_val, hi_out_val = self.qmix.sess.run(
                    [self.qmix.agent_outs, self.qmix.hidden_outs],
                    feed_dict={
                        self.qmix.ph_obs: obs,
                        self.qmix.ph_hidden_states_in: self.qmix.hi_out_val,
                    },
                )

                self.qmix.hi_out_val = hi_out_val

                # print(out_val, hi_out_val)
                print([np.shape(i) for i in (out_val, hi_out_val)])


if __name__ == "__main__":
    unittest.main()
