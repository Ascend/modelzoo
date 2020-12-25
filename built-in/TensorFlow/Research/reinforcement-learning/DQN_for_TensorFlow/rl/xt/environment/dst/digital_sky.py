# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
digital sky game environment
"""
import numpy as np

from gym import spaces
from xt.environment.dst.external_env import ExternalEnv
from xt.environment.dst.state_transform import get_preprocessor
from xt.framework.register import Registers

observation_space = spaces.Dict({
    "self_health": spaces.Box(0, 1000, (1, )),
    "self_shield": spaces.Box(0, 100, (1, )),
    "self_shield_cd": spaces.Box(0, 100, (1, )),
    "self_shield_state": spaces.Box(0, 10, (4, )),
    "self_parry_range": spaces.Box(0, 100, (1, )),
    "self_x": spaces.Box(-np.inf, np.inf, (1, )),
    "self_y": spaces.Box(-np.inf, np.inf, (1, )),
    "self_z": spaces.Box(-np.inf, np.inf, (1, )),
    "self_heading_x": spaces.Box(-np.inf, np.inf, (1, )),
    "self_heading_y": spaces.Box(-np.inf, np.inf, (1, )),
    "self_heading_z": spaces.Box(-np.inf, np.inf, (1, )),
    "self_state": spaces.Discrete(101),
    "self_CurrentHurtCount": spaces.Box(0, 100, (1, )),
    "self_MaxHurtCount": spaces.Box(0, 100, (1, )),
    "self_CurrentParryCountInDefence": spaces.Box(0, 100, (1, )),
    "self_ParryCountInDefence": spaces.Box(0, 100, (1, )),
    "self_Teleport_cd": spaces.Box(0, 100, (1, )),
    "self_in_air": spaces.Discrete(2),
    "opponent_health": spaces.Box(0, 1000, (1, )),
    "opponent_shield": spaces.Box(0, 100, (1, )),
    "opponent_shield_cd": spaces.Box(0, 100, (1, )),
    "opponent_shield_state": spaces.Box(0, 10, (4, )),
    "opponent_parry_range": spaces.Box(0, 100, (1, )),
    "opponent_x": spaces.Box(-np.inf, np.inf, (1, )),
    "opponent_y": spaces.Box(-np.inf, np.inf, (1, )),
    "opponent_z": spaces.Box(-np.inf, np.inf, (1, )),
    "opponent_heading_x": spaces.Box(-np.inf, np.inf, (1, )),
    "opponent_heading_y": spaces.Box(-np.inf, np.inf, (1, )),
    "opponent_heading_z": spaces.Box(-np.inf, np.inf, (1, )),
    "opponent_state": spaces.Discrete(101),
    "opponent_CurrentHurtCount": spaces.Box(0, 200, (1, )),
    "opponent_MaxHurtCount": spaces.Box(0, 100, (1, )),
    "opponent_CurrentParryCountInDefence": spaces.Box(0, 100, (1, )),
    "opponent_ParryCountInDefence": spaces.Box(0, 100, (1, )),
    "opponent_Teleport_cd": spaces.Box(0, 100, (1, )),
    "opponent_in_air": spaces.Discrete(2),
})


@Registers.env.register
class DigitalSky(ExternalEnv):
    """
     DigitalSky server class
    """
    def transfer_state(self, state, *args):
        """
        transform state
        """
        state_dict = {}
        state_dict["self_health"] = np.array([state[0]])
        state_dict["self_shield"] = np.array([state[1]])
        state_dict["self_shield_cd"] = np.array([state[2]])
        state_dict["self_shield_state"] = np.array(state[3:7])
        state_dict["self_parry_range"] = np.array([state[7]])
        state_dict["self_x"] = np.array([state[8]])
        state_dict["self_y"] = np.array([state[9]])
        state_dict["self_z"] = np.array([state[10]])
        state_dict["self_heading_x"] = np.array([state[11]])
        state_dict["self_heading_y"] = np.array([state[12]])
        state_dict["self_heading_z"] = np.array([state[13]])
        state_dict["self_state"] = np.array(state[14])
        state_dict["self_CurrentHurtCount"] = np.array([state[15]])
        state_dict["self_MaxHurtCount"] = np.array([state[16]])
        state_dict["self_CurrentParryCountInDefence"] = np.array([state[17]])
        state_dict["self_ParryCountInDefence"] = np.array([state[18]])
        state_dict["self_Teleport_cd"] = np.array([state[19]])
        state_dict["self_in_air"] = np.array(int(state[20] == 'true'))

        state_dict["opponent_health"] = np.array([state[21]])
        state_dict["opponent_shield"] = np.array([state[22]])
        state_dict["opponent_shield_cd"] = np.array([state[23]])
        state_dict["opponent_shield_state"] = np.array(state[24:28])
        state_dict["opponent_parry_range"] = np.array([state[28]])
        state_dict["opponent_x"] = np.array([state[29]])
        state_dict["opponent_y"] = np.array([state[30]])
        state_dict["opponent_z"] = np.array([state[31]])
        state_dict["opponent_heading_x"] = np.array([state[32]])
        state_dict["opponent_heading_y"] = np.array([state[33]])
        state_dict["opponent_heading_z"] = np.array([state[34]])
        state_dict["opponent_state"] = np.array(state[35])
        state_dict["opponent_CurrentHurtCount"] = np.array([state[36]])
        state_dict["opponent_MaxHurtCount"] = np.array([state[37]])
        state_dict["opponent_CurrentParryCountInDefence"] = np.array([state[38]])
        state_dict["opponent_ParryCountInDefence"] = np.array([state[39]])
        state_dict["opponent_Teleport_cd"] = np.array([state[40]])
        state_dict["opponent_in_air"] = np.array(int(state[41] == 'true'))

        processor = get_preprocessor(observation_space)(observation_space)
        state = processor.transform(state_dict)
        return state
