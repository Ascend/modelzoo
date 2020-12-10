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
"""Test function for vega.run"""
import unittest


def lazy(func):
    """lazy function wrapper

    :param func: function name
    """
    attr_name = "_lazy_" + func.__name__

    def lazy_func(*args, **kwargs):
        """Wrapper of lazy func

        :param args: any object
        :param kwargs: any kwargs
        :return:
        """
        if not hasattr(func, attr_name):
            setattr(func, attr_name, func(*args, **kwargs))
        return getattr(func, attr_name)

    return lazy_func


@lazy
def env_args(args):
    """A lazy function will be execute when call

    :param args: any object
    :return:
    """
    return args


class TestPipeline(unittest.TestCase):
    """Test lazy function worked in pipeline"""

    def test_env_args(self):
        """Test function 'env_args' is a lazy function"""
        args = {'env': 'test'}
        env_args(args)
        self.assertEqual(env_args(), {'env': 'test'})
