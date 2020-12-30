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

To view the actual content, go to: flow/networks/figure_eight.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.figure_eight import FigureEightNetwork
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.figure_eight',
            'flow.networks.figure_eight.FigureEightNetwork')
class FigureEightScenario(FigureEightNetwork):
    """See parent class."""

    pass


@deprecated('flow.scenarios.figure_eight',
            'flow.networks.figure_eight.FigureEightNetwork')
class Figure8Scenario(FigureEightNetwork):
    """See parent class."""

    pass