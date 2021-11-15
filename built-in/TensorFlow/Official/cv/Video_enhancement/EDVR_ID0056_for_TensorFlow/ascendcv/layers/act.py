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


def ReLU(x, name=None):
    x = tf.nn.relu(x, name=name)

    return x


def LeakyReLU(x, alpha=0.1, name=None):
    x = tf.nn.leaky_relu(x, alpha=alpha, name=name)

    return x


class ActLayer(object):

    def __init__(self, cfg, name=None):
        super(ActLayer, self).__init__()
        self.type = cfg.get('type').lower()
        if self.type == 'leakyrelu':
            self.alpha = cfg.get('alpha', 0.2)
        self.name = name

    def _forward(self, x):
        if self.type == 'relu':
            return ReLU(x, name=self.name)
        elif self.type == 'leakyrelu':
            return LeakyReLU(x, alpha=self.alpha, name=self.name)
        else:
            raise NotImplementedError

    def __call__(self, x):
        # shape = list(map(int, x.shape))
        shape = x.get_shape().as_list()
        if len(shape) == 5:
            # TODO
            # Ascend currently do not support 5D relu
            x_4d = tf.reshape(x, [-1] + shape[2:])
            x_4d = self._forward(x_4d)
            x = tf.reshape(x_4d, shape)
        else:
            x = self._forward(x)

        return x
