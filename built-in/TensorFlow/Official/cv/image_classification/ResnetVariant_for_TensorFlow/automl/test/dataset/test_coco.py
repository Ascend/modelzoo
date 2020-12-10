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
"""This is an test to copy."""
import unittest
import copy
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakeTFPipeStep(PipeStep, unittest.TestCase):
    """Fake TF PipeStep."""

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        """Do train."""
        data_cls = ClassFactory.get_cls(ClassType.DATASET)
        data_cfg = copy.deepcopy(ClassFactory.__configs__.get(ClassType.DATASET))
        data_cfg.pop('type')
        train_data, valid_data = [
            data_cls(**data_cfg, mode=mode) for mode in ['train', 'val']
        ]


class TestDataset(unittest.TestCase):
    """Test Dataset."""

    def test_coco(self):
        """Test coco."""
        vega.run('./coco_tf.yml')


if __name__ == "__main__":
    unittest.main()
