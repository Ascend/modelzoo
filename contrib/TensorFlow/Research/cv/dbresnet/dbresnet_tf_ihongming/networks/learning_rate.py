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
from __future__ import division

import os
import sys
import pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))

import numpy as np
import math
import tensorflow as tf
from config.db_config import cfg


def learning_rate_with_decay(start_lr=0.007, power=0.9):
    total_stpes = cfg.TRAIN.MAX_STEPS
    steps_per_epoch = 1255//8
    
    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step//steps_per_epoch, tf.float32)   
        rate = tf.math.pow((1.0 - (global_step / 1201)), power)
        return tf.cast(start_lr, tf.float32) * rate

    return learning_rate_fn


def learning_rate_step_decay(start_lr=0.007, power=0.9):
    total_stpes = cfg.TRAIN.MAX_STEPS
    
    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.float32)
        rate = tf.math.pow((1.0 - (global_step / total_stpes)), power)
        return tf.cast(start_lr, tf.float32) * rate

    return learning_rate_fn


def learning_rate_with_exponential_decay():
    lr = cfg.TRAIN.LEARNING_RATE
    ds = cfg.ADAM_DECAY_STEP
    dr = cfg.ADAM_DECAY_RATE

    def learning_rate_fn(global_step):
        return tf.train.exponential_decay(lr, global_step, decay_steps=ds, decay_rate=dr, staircase=True)
    
    return learning_rate_fn


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tqdm
    
    init = tf.global_variables_initializer()
    lrs = []

    with tf.Session() as sess:
        sess.run(init)
        learning_rate_fn = learning_rate_with_decay()
        for global_step in tqdm.tqdm(range(1200)):  
            lr,epo,rate = sess.run(learning_rate_fn(global_step * 156))
            lrs.append(lr)
            print(epo,"_______",lr,"_______",rate)

        plt.plot(range(1200), lrs, color="r", linestyle="-", linewidth=1)
        plt.savefig("test.png", dpi=120)
    
