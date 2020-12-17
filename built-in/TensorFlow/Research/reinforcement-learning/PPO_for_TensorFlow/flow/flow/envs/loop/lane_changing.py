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

To view the actual content, go to: flow/envs/ring/lane_change_accel.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv as LCEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelPOEnv as LCPOEnv


@deprecated('flow.envs.loop.lane_changing',
            'flow.envs.ring.lane_change_accel.LaneChangeAccelEnv')
class LaneChangeAccelEnv(LCEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.loop.lane_changing',
            'flow.envs.ring.lane_change_accel.LaneChangeAccelPOEnv')
class LaneChangeAccelPOEnv(LCPOEnv):
    """See parent class."""

    pass
