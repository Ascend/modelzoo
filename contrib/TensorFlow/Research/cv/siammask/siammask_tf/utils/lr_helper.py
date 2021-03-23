# --------------------------------------------------------
# 学习率lr更新
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
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
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

# --------------------------------------------------------
from __future__ import division
import numpy as np
import math
import tensorflow as tf


def learning_rate_with_decay_64(epochs=20,
                                batch_size=64, boundary_epochs=np.arange(14) + 6, batch_denom=64, start_lr=0.005,
                                end_lr=5e-5, num_images=600000,
                                base_lr=0.001, highest_lr=0.005, warmup=True):
    batches_per_epoch = num_images / batch_size

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = np.logspace(math.log10(start_lr), math.log10(end_lr), epochs - 5).tolist()

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (base_lr + (
                    (highest_lr - base_lr) * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn


def learning_rate_with_decay_16(epochs=20,
                                batch_size=16, boundary_epochs=np.arange(14) + 6, batch_denom=64, start_lr=0.0025,
                                end_lr=2.5e-5, num_images=600000,
                                base_lr=0.0005, highest_lr=0.0025, warmup=True):
    batches_per_epoch = num_images / batch_size

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = np.logspace(math.log10(start_lr), math.log10(end_lr), epochs - 5).tolist()

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (base_lr + (
                    (highest_lr - base_lr) * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import tqdm

    init = tf.global_variables_initializer()
    lrs = []

    with tf.Session() as sess:
        sess.run(init)
        learning_rate_fn = learning_rate_with_decay_64()
        for global_step in tqdm.tqdm(range(20)):  # 用局部的global_step代替
            lr = sess.run(learning_rate_fn(global_step * 37500))
            lrs.append(lr)
        plt.plot(range(10), lrs, color="r", linestyle="-", linewidth=1)
        plt.savefig("test.png", dpi=120)
