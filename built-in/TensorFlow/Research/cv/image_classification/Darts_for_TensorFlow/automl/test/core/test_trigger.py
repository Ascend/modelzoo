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
"""Test function for vega.core.trainer.trigger."""
import unittest

from vega.core.trainer.trigger import Trigger


@Trigger.activate('train')
def train():
    """Activate log trigger."""
    print("train")


@Trigger.activate('valid')
def valid():
    """Activate valid trigger."""
    print("valid")


@Trigger.register(['train', 'valid'])
class LoggerTrigger(Trigger):
    """Register logger trigger."""

    def before(self):
        """Execute before function."""
        print("Before train")

    def after(self):
        """Execute after function"""
        print("After train")


@Trigger.register('train')
class ProfileTrigger(Trigger):
    """Register Profile trigger."""

    def before(self):
        """Execute before function."""
        print("Profile: Before train")

    def after(self):
        """Execute after function"""
        print("Profile: After train")


class TestTrigger(unittest.TestCase):
    """Test trigger wrapper."""

    def test_trigger(self):
        """Test trigger"""
        train()
        valid()
        print(Trigger.__triggers__)


if __name__ == "__main__":
    unittest.main()
