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
"""Pending deprecation file.

To view the actual content, go to: flow/envs/bottleneck.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.bottleneck import BottleneckEnv as BEnv
from flow.envs.bottleneck import BottleneckAccelEnv as BAEnv
from flow.envs.bottleneck import BottleneckDesiredVelocityEnv as BDVEnv


@deprecated('flow.envs.bottleneck_env',
            'flow.envs.bottleneck.BottleneckEnv')
class BottleneckEnv(BEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.bottleneck_env',
            'flow.envs.bottleneck.BottleneckAccelEnv')
class BottleNeckAccelEnv(BAEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.bottleneck_env',
            'flow.envs.bottleneck.BottleneckDesiredVelocityEnv')
class DesiredVelocityEnv(BDVEnv):
    """See parent class."""

    pass
