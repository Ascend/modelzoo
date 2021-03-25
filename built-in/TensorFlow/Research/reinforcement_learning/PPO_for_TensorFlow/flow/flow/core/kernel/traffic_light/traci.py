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
"""Script containing the TraCI traffic light kernel class."""

from flow.core.kernel.traffic_light import KernelTrafficLight
import traci.constants as tc


class TraCITrafficLight(KernelTrafficLight):
    """Sumo traffic light kernel.

    Implements all methods discussed in the base traffic light kernel class.
    """

    def __init__(self, master_kernel):
        """Instantiate the sumo traffic light kernel.

        Parameters
        ----------
        master_kernel : flow.core.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        KernelTrafficLight.__init__(self, master_kernel)

        self.__tls = dict()  # contains current time step traffic light data
        self.__tls_properties = dict()  # traffic light xml properties

        # names of nodes with traffic lights
        self.__ids = []

        # number of traffic light nodes
        self.num_traffic_lights = 0

    def pass_api(self, kernel_api):
        """See parent class.

        Subscriptions and vehicle IDs are also added here.
        """
        KernelTrafficLight.pass_api(self, kernel_api)

        # names of nodes with traffic lights
        self.__ids = kernel_api.trafficlight.getIDList()

        # number of traffic light nodes
        self.num_traffic_lights = len(self.__ids)

        # subscribe the traffic light signal data
        for node_id in self.__ids:
            self.kernel_api.trafficlight.subscribe(
                node_id, [tc.TL_RED_YELLOW_GREEN_STATE])

    def update(self, reset):
        """See parent class."""
        tls_obs = {}
        for tl_id in self.__ids:
            tls_obs[tl_id] = \
                self.kernel_api.trafficlight.getSubscriptionResults(tl_id)
        self.__tls = tls_obs.copy()

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def set_state(self, node_id, state, link_index="all"):
        """See parent class."""
        if link_index == "all":
            # if lights on all lanes are changed
            self.kernel_api.trafficlight.setRedYellowGreenState(
                tlsID=node_id, state=state)
        else:
            # if lights on a single lane is changed
            self.kernel_api.trafficlight.setLinkState(
                tlsID=node_id, tlsLinkIndex=link_index, state=state)

    def get_state(self, node_id):
        """See parent class."""
        return self.__tls[node_id][tc.TL_RED_YELLOW_GREEN_STATE]
