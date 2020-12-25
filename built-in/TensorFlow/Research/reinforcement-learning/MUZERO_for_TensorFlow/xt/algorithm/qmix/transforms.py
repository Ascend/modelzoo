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
"""transform utils for qmix algorithm."""
import numpy as np

import tensorflow as tf


class Transform:
    """transform base class"""
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHotTf(Transform):
    """transform with tensorflow"""
    def __init__(self, out_dim, dtype=tf.float32):
        self.out_dim = out_dim
        self.dtype = dtype

    def transform(self, tensor):
        tensor_indices = np.squeeze(tensor, axis=-1)
        one_hot = tf.one_hot(
            indices=tensor_indices,
            depth=self.out_dim,
            on_value=1.0,
            off_value=0.0,
            axis=-1,
            dtype=self.dtype,
        )
        return one_hot

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), self.dtype


class OneHotNp(Transform):
    """transform with numpy"""
    def __init__(self, out_dim, dtype=np.float):
        self.out_dim = out_dim
        self.dtype = dtype

    def transform(self, tensor):
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        # print(np.array(targets).reshape(-1))
        res = np.eye(self.out_dim)[tensor.reshape(-1)]
        targets = res.reshape([*(tensor.shape[:-1]), self.out_dim])
        return targets.astype(self.dtype)

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), self.dtype


def test_func():
    """check with func between numpy and tf."""
    output_dim = 11

    data = [[[[10], [2], [5], [2], [10]]]]

    # # tf
    # oh = OneHotTf(output_dim)
    # r = oh.transform(a)
    # with tf.Session() as sess:
    #     r = sess.run(r)
    #     print(r, r.shape)

    # np
    np_onehot = OneHotNp(output_dim)
    np_ret = np_onehot.transform(data)
    print(np_ret)


if __name__ == "__main__":
    test_func()
