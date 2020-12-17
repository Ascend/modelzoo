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
"""Contains the RLController class."""

from flow.controllers.base_controller import BaseController


class RLController(BaseController):
    """RL Controller.

    Vehicles with this class specified will be stored in the list of the RL IDs
    in the Vehicles class.

    Usage: See base class for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification

    Examples
    --------
    A set of vehicles can be instantiated as RL vehicles as follows:

        >>> from flow.core.params import VehicleParams
        >>> vehicles = VehicleParams()
        >>> vehicles.add(acceleration_controller=(RLController, {}))

    In order to collect the list of all RL vehicles in the next, run:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> rl_ids = env.k.vehicle.get_rl_ids()
    """

    def __init__(self, veh_id, car_following_params):
        """Instantiate an RL Controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params)
