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
import random
import unittest

from vega.algorithms.hpo.asha_hpo import AshaHpo
from vega.algorithms.nas import BackboneNas
from vega.core.common.config import Config
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm


def run(desc):
    """Run."""
    desc.update({'performance': random.randint(0, 10)})
    return desc


class TestSearchAlgorithm(unittest.TestCase):
    """Test Search Algorithm."""

    def test_backbone_nas(self):
        """Test backbone nas."""
        search_space = dict(type='SearchSpace', modules=['backbone', 'head'],
                            head=dict(LinearClassificationHead=dict(num_classes=[10])),
                            backbone=dict(
                                ResNetVariant=dict(base_depth=[18, 34, 50, 101], base_channel=[32, 48, 56, 64],
                                                   doublechannel=[3, 4], downsample=[3, 4])))
        search_space = Config(dict(search_space=search_space))
        backbone_nas1 = SearchAlgorithm(search_space, type='BackboneNas',
                                        policy=Config(random_ratio=0.55, num_mutate=50))
        self.assertEqual(backbone_nas1.config.policy.random_ratio, 0.55)

        backbone_nas2 = BackboneNas(search_space)
        self.assertEqual(backbone_nas2.config.policy.random_ratio, 0.2)

    def test_asha(self):
        """Test asha."""
        searchspace = Config({"search_space": {"type": 'SearchSpace', "hyperparameters": [
            {"key": "name", "type": "INT_CAT", "range": [8, 16, 32, 64, 128, 256]}]}})
        asha = AshaHpo(searchspace, policy=dict(config_count=10))
        while not asha.is_completed:
            desc = asha.search()
            print(desc)
            perf = run(desc)
            print(perf)
            asha.update(perf)


if __name__ == "__main__":
    unittest.main()
