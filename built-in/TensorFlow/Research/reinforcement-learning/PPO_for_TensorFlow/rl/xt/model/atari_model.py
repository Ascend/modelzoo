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
"""model architecture for atari game"""


def get_atari_filter(shape):
    """get default model set for atari environments"""
    shape = list(shape)
    # (out_size, kernel, stride)
    filters_84x84 = [
        [16, 8, 4],
        [32, 4, 2],
        [256, 11, 1],
    ]
    filters_42x42 = [
        [16, 4, 2],
        [32, 4, 2],
        [256, 11, 1],
    ]
    if len(shape) == 3 and shape[:2] == [84, 84]:
        return filters_84x84
    elif len(shape) == 3 and shape[:2] == [42, 42]:
        return filters_42x42
    else:
        raise ValueError("Without default architecture for obs shape {}".format(shape))
