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

import tensorflow as tf
import numpy as np

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def test():
    batch_size = 10
    depth = 128
    output_dim = 100

    inputs = tf.Variable(tf.random_normal([batch_size, depth]))
    previous_state = tf.Variable(tf.random_normal([batch_size, output_dim]))  # 前一个状态的输出
    gruCell = tf.nn.rnn_cell.GRUCell(output_dim)
    # gruCell = tf.keras.layers.GRUCell(output_dim)

    output, state = gruCell(inputs, previous_state)
    # print(output)
    # print("state, ", state)

    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        # print(sess.run(inputs))
        out = sess.run(output)
        print(np.shape(out))
        sta = sess.run(state)
        print(np.shape(sta))

        print((np.array(out) == np.array(sta)).all())


if __name__ == "__main__":
    test()
