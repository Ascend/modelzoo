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
import os
from roma.env import init_env
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        self.assertEqual(self.cfg.task.local_base_path, "/efs/cache/local/")
        dataset = ClassFactory.get_cls("dataset")
        train_dataset = dataset(mode='train')
        self.assertEqual(len(train_dataset), 25000)


class TestDataset(unittest.TestCase):

    def test_roma(self):
        init_env('hb1_y')
        vega.run('./roma_config.yml')


if __name__ == "__main__":
    unittest.main()
