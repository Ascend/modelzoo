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
import torchvision.transforms as tf
from vega.datasets.common.dataset import Dataset
from roma.env import init_env
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.task_ops import TaskOps
from vega.core.common.file_ops import FileOps
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, TaskOps, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        FileOps.copy_folder("s3://automl-hn1/liuzhicheng/test_roma/", "/cache/test/")
        test_file = len([x for x in os.listdir(os.path.dirname("/cache/test/"))])
        self.assertEqual(test_file, 3)
        self.assertEqual(self.local_base_path, "/efs/{}".format(self.task_id))
        self.assertEqual(self.output_subpath, "output/")
        self.assertEqual(self.get_worker_subpath("1", "10"), "workers/1/10/")


class TestDataset(unittest.TestCase):

    def test_roma(self):
        init_env()
        vega.run('./roma.yml')


if __name__ == "__main__":
    unittest.main()
