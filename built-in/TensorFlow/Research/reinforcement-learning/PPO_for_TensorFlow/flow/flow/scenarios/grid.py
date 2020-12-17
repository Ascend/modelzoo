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

To view the actual content, go to: flow/networks/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.networks.traffic_light_grid import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.grid',
            'flow.networks.traffic_light_grid.TrafficLightGridNetwork')
class SimpleGridScenario(TrafficLightGridNetwork):
    """See parent class."""

    pass
