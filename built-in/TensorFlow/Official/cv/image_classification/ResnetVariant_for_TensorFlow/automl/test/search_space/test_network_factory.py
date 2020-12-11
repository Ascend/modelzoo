# coding=utf-8
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
import unittest
import torch.nn as nn
from vega.search_space.networks import NetTypes, NetTypesMap, Network, NetworkFactory


@NetworkFactory.register(NetTypes.BACKBONE)
class SimpleNetwork(Network):

    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.net = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.net(x)


class TestNetworkFactory(unittest.TestCase):

    def test_simple_network(self):
        module_type = NetTypesMap['backbone']
        net_class = NetworkFactory.get_network(module_type, 'SimpleNetwork')
        self.assertEqual(isinstance(net_class(), SimpleNetwork), True)


if __name__ == "__main__":
    unittest.main()
