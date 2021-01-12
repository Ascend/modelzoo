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

from collections import OrderedDict

import tensorflow as tf

from tensorflow.keras.applications.vgg19 import VGG19


def scale_initialization(weights, FLAGS):
    return [
        tf.assign(weight, weight * FLAGS.weight_initialize_scale)
        for weight in weights
    ]


def _transfer_vgg19_weight(FLAGS, weight_dict):
    from_model = VGG19(include_top=False,
                       weights=FLAGS.VGG19_weights,
                       input_tensor=None,
                       input_shape=(FLAGS.HR_image_size, FLAGS.HR_image_size,
                                    FLAGS.channel))

    fetch_weight = []

    for layer in from_model.layers:
        if 'conv' in layer.name:
            W, b = layer.get_weights()

            fetch_weight.append(
                tf.assign(
                    weight_dict[
                        'loss_generator/perceptual_vgg19/{}/kernel'.format(
                            layer.name)], W))
            fetch_weight.append(
                tf.assign(
                    weight_dict[
                        'loss_generator/perceptual_vgg19/{}/bias'.format(
                            layer.name)], b))

    return fetch_weight


def load_vgg19_weight(FLAGS):
    vgg_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='loss_generator/perceptual_vgg19')

    assert len(
        vgg_weight
    ) > 0, 'No VGG19 weight was collected. The target scope might be wrong.'

    weight_dict = {}
    for weight in vgg_weight:
        weight_dict[weight.name.rsplit(':', 1)[0]] = weight

    return _transfer_vgg19_weight(FLAGS, weight_dict)


def extract_weight(network_vars):
    weight_dict = OrderedDict()

    for weight in network_vars:
        weight_dict[weight.name] = weight.eval()

    return weight_dict


def interpolate_weight(FLAGS, pretrain_weight):
    fetch_weight = []
    alpha = FLAGS.interpolation_param

    for name, pre_weight in pretrain_weight.items():
        esrgan_weight = tf.get_default_graph().get_tensor_by_name(name)

        assert pre_weight.shape == esrgan_weight.shape, 'The shape of weights does not match'

        fetch_weight.append(
            tf.assign(esrgan_weight,
                      (1 - alpha) * pre_weight + alpha * esrgan_weight))

    return fetch_weight
