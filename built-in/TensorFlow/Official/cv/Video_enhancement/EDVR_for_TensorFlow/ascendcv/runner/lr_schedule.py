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
import math

import tensorflow as tf


class BaseSchedule(object):

    def __init__(self, base_lr):
        self.lr = tf.placeholder(tf.float32, shape=[], name='leraning_rate')
        self.cur_step = 0
        self.base_lr = base_lr
        self.cur_lr = base_lr

    def __call__(self):
        self.cur_step += 1
        return self.cur_lr


class CosineSchedule(BaseSchedule):

    def __init__(self, base_lr, total_steps, min_lr):
        super(CosineSchedule, self).__init__(base_lr)
        self.total_steps = total_steps[0]
        self.min_lr = min_lr
        self.alpha = min_lr / base_lr

    def __call__(self):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.cur_step / self.total_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        self.cur_lr = self.base_lr * decayed
        return super(CosineSchedule, self).__call__()


class RestartCosineSchedule(BaseSchedule):

    def __init__(self, base_lr, total_steps, restart_weights, min_lr):
        super(RestartCosineSchedule, self).__init__(base_lr)
        self.total_steps = total_steps
        self.restart_weights = restart_weights
        self.min_lr = min_lr
        self.alpha = min_lr / base_lr

    def _match_stage(self):
        cur_step = self.cur_step
        for total_steps, restart_weight in zip(self.total_steps, self.restart_weights):
            if cur_step < total_steps:
                return cur_step, total_steps, self.base_lr * restart_weight
            else:
                cur_step -= total_steps
        raise ValueError('Should have stopped')

    def __call__(self):
        cur_step, total_steps, base_lr = self._match_stage()
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / total_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        self.cur_lr = base_lr * decayed
        return super(RestartCosineSchedule, self).__call__()


def build_schedule(lr_cfg):

    lr_type = lr_cfg.type.lower()
    base_lr = lr_cfg.base_lr
    total_steps = lr_cfg.total_steps

    if lr_type == 'cosine':
        min_lr = lr_cfg.min_lr
        return CosineSchedule(base_lr, total_steps, min_lr)
    elif lr_type == 'cosinerestart':
        min_lr = lr_cfg.min_lr
        restart_weights = lr_cfg.restart_weights
        return RestartCosineSchedule(base_lr, total_steps, restart_weights, min_lr)
    elif lr_type == 'step':
        raise NotImplementedError
    else:
        raise KeyError('Unkown type {}'.format(lr_type))
