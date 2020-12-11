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
"""Test Metrics Wrapper."""

import unittest
from inspect import signature as sig
from functools import wraps


def metric(name=None):
    """Make function as a metrics, use the same params from configuration.

    :param func: source function
    :return: wrapper
    """

    def decorator(func):
        """Provide input param to decorator.

        :param func: wrapper function
        :return: decoratpr
        """
        setattr(func, 'name', name or func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Make function as a wrapper."""
            params_sig = sig(func).parameters
            params = {param: value for param, value in kwargs.items() if param in params_sig}
            return func(*args, **params)

        return wrapper

    return decorator


@metric('addx')
def add(x, y):
    """Add x and y."""
    print(x, y)
    return x + y


class TestMetric(unittest.TestCase):
    """Set metrics wrapper to function, merge config with input params and specify metric name."""

    def test_metric(self):
        """Test metric name and params worked"""
        self.assertEqual(add.name, 'addx')
        self.assertEqual(add.__name__, 'add')
        self.assertEqual(add(1, 2), 3)
