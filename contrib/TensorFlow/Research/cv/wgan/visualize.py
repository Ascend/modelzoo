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
# Copyright 2021 Huawei Technologies Co., Ltd
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

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def split(x):
    assert type(x) == int
    t = int(np.floor(np.sqrt(x)))
    for a in range(t, 0, -1):
        if x % a == 0:
            return a, x / a


def grid_transform(x, size):
    a, b = split(x.shape[0])
    h, w, c = size[0], size[1], size[2]
    a = int(a)
    b = int(b)
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x


def grid_show(fig, x, size):
    ax = fig.add_subplot(111)
    x = grid_transform(x, size)
    if len(x.shape) > 2:
        ax.imshow(x)
    else:
        ax.imshow(x, cmap='gray')
