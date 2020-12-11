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
"""Test function for `vega.core.common.file_ops.copy`"""
import unittest
import os
from vega.core.common import FileOps


class TestDataset(unittest.TestCase):

    def test_copy(self):
        file_dir = os.path.abspath(os.path.dirname(__file__))
        new_dir = file_dir + '/automl'
        os.mkdir(new_dir)
        open(os.path.join(file_dir, 'test1.txt'), 'a').close()
        src_file = os.path.join(file_dir, 'test1.txt')
        dst_file = os.path.join(new_dir, "test1.txt")
        FileOps.copy_file(src_file, dst_file)
        self.assertEqual(os.path.isfile(dst_file), True)
        new_folder = file_dir + "/new"
        FileOps.copy_folder(new_dir, new_folder)
        file_num = len([x for x in os.listdir(new_folder)])
        self.assertEqual(file_num, 1)


if __name__ == "__main__":
    unittest.main()
