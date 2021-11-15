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
import numbers
import tensorflow as tf


def tf_slicing(x, axis, slice_range, keep_dim=False):
    target_dim = int(x.shape[axis])
    ndim = len(x.shape)
    begin = list(ndim * [0])
    size = x.get_shape().as_list()

    if isinstance(slice_range, (list, tuple)):
        begin[axis] = slice_range[0]
        size[axis] = slice_range[1] - slice_range[0]
    elif isinstance(slice_range, numbers.Integral):
        begin[axis] = slice_range
        size[axis] = 1
    else:
        raise ValueError

    x_slice = tf.slice(x, begin, size)
    if size[axis] == 1 and not keep_dim:
        x_slice = tf.squeeze(x_slice, axis)

    return x_slice


def tf_split(x, num_or_size_splits, axis=0, num=None, keep_dims=False):
    x_list = tf.split(x, num_or_size_splits, axis, num)

    if not keep_dims:
        x_list2 = [tf.squeeze(x_, axis) for x_ in x_list]
        return x_list2

    return x_list

