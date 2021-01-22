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

import tensorflow as tf
def depth_to_space(x, scale, use_default=False):
    # Ascend implementation of tf.depth_to_space is not accurate so far
    # Thanks to Huang Wei h00573990
    if use_default:
        out = tf.depth_to_space(x, scale)
    else:
        # b, h, w, c = list(map(int, x.shape))
        b, h, w, c = x.get_shape().as_list()
        c_scaled = c // (scale**2)
        out = tf.reshape(x, [-1, h, w, scale, scale, c_scaled])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, [-1, h * scale, w * scale, c_scaled])
    return out


def resize(x, size, align_corners=False, name=None, half_pixel_centers=False, method='bicubic'):
    if method == 'bicubic':
        upsampling = tf.image.resize_bicubic
    elif method == 'bilinear':
        upsampling = tf.image.resize_bilinear
    else:
        raise ValueError
    return upsampling(x, size=size, align_corners=align_corners, name=name, half_pixel_centers=half_pixel_centers)

