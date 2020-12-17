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
"""Contains the I-210 sub-network class."""

from flow.networks.base import Network

EDGES_DISTRIBUTION = [
    # Main highway
    "119257914",
    "119257908#0",
    "119257908#1-AddedOnRampEdge",
    "119257908#1",
    "119257908#1-AddedOffRampEdge",
    "119257908#2",
    "119257908#3",

    # On-ramp
    "27414345",
    "27414342#0",
    "27414342#1-AddedOnRampEdge",

    # Off-ramp
    "173381935",
]


class I210SubNetwork(Network):
    """A network used to simulate the I-210 sub-network.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import I210SubNetwork
    >>>
    >>> network = I210SubNetwork(
    >>>     name='I-210_subnetwork',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams()
    >>> )
    """

    def specify_routes(self, net_params):
        """See parent class.

        Routes for vehicles moving through the bay bridge from Oakland to San
        Francisco.
        """
        rts = {
            # Main highway
            "119257914": [
                (["119257914", "119257908#0", "119257908#1-AddedOnRampEdge",
                  "119257908#1", "119257908#1-AddedOffRampEdge", "119257908#2",
                  "119257908#3"],
                 1),  # HOV: 1509 (on ramp: 57), Non HOV: 6869 (onramp: 16)
                # (["119257914", "119257908#0", "119257908#1-AddedOnRampEdge",
                #   "119257908#1", "119257908#1-AddedOffRampEdge", "173381935"],
                #  17 / 8378)
            ],
            # "119257908#0": [
            #     (["119257908#0", "119257908#1-AddedOnRampEdge", "119257908#1",
            #       "119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      1.0),
            #     # (["119257908#0", "119257908#1-AddedOnRampEdge", "119257908#1",
            #     #   "119257908#1-AddedOffRampEdge", "173381935"],
            #     #  0.5),
            # ],
            # "119257908#1-AddedOnRampEdge": [
            #     (["119257908#1-AddedOnRampEdge", "119257908#1",
            #       "119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      1.0),
            #     # (["119257908#1-AddedOnRampEdge", "119257908#1",
            #     #   "119257908#1-AddedOffRampEdge", "173381935"],
            #     #  0.5),
            # ],
            # "119257908#1": [
            #     (["119257908#1", "119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      1.0),
            #     # (["119257908#1", "119257908#1-AddedOffRampEdge", "173381935"],
            #     #  0.5),
            # ],
            # "119257908#1-AddedOffRampEdge": [
            #     (["119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      1.0),
            #     # (["119257908#1-AddedOffRampEdge", "173381935"],
            #     #  0.5),
            # ],
            # "119257908#2": [
            #     (["119257908#2", "119257908#3"], 1),
            # ],
            # "119257908#3": [
            #     (["119257908#3"], 1),
            # ],
            #
            # # On-ramp
            # "27414345": [
            #     (["27414345", "27414342#1-AddedOnRampEdge",
            #       "27414342#1",
            #       "119257908#1-AddedOnRampEdge", "119257908#1",
            #       "119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      1 - 9 / 321),
            #     (["27414345", "27414342#1-AddedOnRampEdge",
            #       "27414342#1",
            #       "119257908#1-AddedOnRampEdge", "119257908#1",
            #       "119257908#1-AddedOffRampEdge", "173381935"],
            #      9 / 321),
            # ],
            # "27414342#0": [
            #     (["27414342#0", "27414342#1-AddedOnRampEdge",
            #       "27414342#1",
            #       "119257908#1-AddedOnRampEdge", "119257908#1",
            #       "119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      1 - 20 / 421),
            #     (["27414342#0", "27414342#1-AddedOnRampEdge",
            #       "27414342#1",
            #       "119257908#1-AddedOnRampEdge", "119257908#1",
            #       "119257908#1-AddedOffRampEdge", "173381935"],
            #      20 / 421),
            # ],
            # "27414342#1-AddedOnRampEdge": [
            #     (["27414342#1-AddedOnRampEdge", "27414342#1", "119257908#1-AddedOnRampEdge",
            #       "119257908#1", "119257908#1-AddedOffRampEdge", "119257908#2",
            #       "119257908#3"],
            #      0.5),
            #     (["27414342#1-AddedOnRampEdge", "27414342#1", "119257908#1-AddedOnRampEdge",
            #       "119257908#1", "119257908#1-AddedOffRampEdge", "173381935"],
            #      0.5),
            # ],
            #
            # # Off-ramp
            # "173381935": [
            #     (["173381935"], 1),
            # ],
        }

        return rts
