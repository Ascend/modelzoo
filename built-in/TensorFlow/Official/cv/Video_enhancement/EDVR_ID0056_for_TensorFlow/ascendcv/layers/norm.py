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


def BatchNorm(x, momentum=0.999, is_train=True, name='BatchNorm'):
    output = tf.layers.batch_normalization(x, momentum=momentum, epsilon=1e-3, name=name, training=is_train)

    return output


class NormLayer(object):

    def __init__(self, cfg, is_train, name=None):
        super(NormLayer, self).__init__()
        self.type = cfg.get('type').lower()
        self.is_train = is_train
        self.name = name

    def _forward(self, x):
        if self.type == 'bn':
            return BatchNorm(x, is_train=self.is_train, name=self.name)
        else:
            raise NotImplementedError

    def __call__(self, x):
        # shape = list(map(int, x.shape))
        shape = x.get_shape().as_list()
        if len(shape) == 5:
            # TODO
            # Ascend currently do not support 5D bn
            x_4d = tf.reshape(x, [-1] + shape[2:])
            x_4d = self._forward(x_4d)
            x = tf.reshape(x_4d, shape)
        else:
            x = self._forward(x)

        return x
