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
"""Contains the base routing controller class."""


class BaseRouter:
    """Base class for routing controllers.

    These controllers are used to dynamically change the routes of vehicles
    after initialization.

    Usage
    -----
    >>> from flow.core.params import VehicleParams
    >>> from flow.controllers import ContinuousRouter
    >>> vehicles = VehicleParams()
    >>> vehicles.add("human", routing_controller=(ContinuousRouter, {}))

    Note: You can replace "ContinuousRouter" with any routing controller you
    want.

    Parameters
    ----------
    veh_id : str
        ID of the vehicle this controller is used for
    router_params : dict
        Dictionary of router params
    """

    def __init__(self, veh_id, router_params):
        """Instantiate the base class for routing controllers."""
        self.veh_id = veh_id
        self.router_params = router_params

    def choose_route(self, env):
        """Return the routing method implemented by the controller.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base.py

        Returns
        -------
        list or None
            The sequence of edges the vehicle should adopt. If a None value
            is returned, the vehicle performs no routing action in the current
            time step.
        """
        raise NotImplementedError
