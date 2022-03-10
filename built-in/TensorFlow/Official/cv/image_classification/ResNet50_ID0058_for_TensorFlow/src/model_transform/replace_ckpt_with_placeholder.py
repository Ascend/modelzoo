# coding=utf-8
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

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import argparse

sys.path.append("../")

from src.models.resnet50 import res50_model as ml
from src.configs.res50_256bs_1p import res50_config

config = res50_config()
tf.reset_default_graph()


def add_placeholder_on_ckpt(input_file, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = ml.Model(config, None, None, None, None, None, None)
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [1,
                                             config.get("height"),
                                             config.get("width"),
                                             3])

        with tf.variable_scope('fp32_vars'):
            model_func = model.get_model_func()
            top_layer = model_func(
                inputs, data_format=config['data_format'],
                training=False,
                conv_initializer=config['conv_init'],
                bn_init_mode=config['bn_init_mode'],
                bn_gamma_initial_value=config['bn_gamma_initial_value'])

        saver = tf.train.Saver(var_list=tf.global_variables(scope='fp32_vars'))
        sess.run(tf.global_variables_initializer())
        saver_to_restore = tf.train.Saver()
        saver_to_restore.restore(sess, input_file)
        saver.save(sess, save_path=save_path)
        print('TensorFlow model checkpoint has been saved to {}'.format(
            save_path))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='the input file name')
    parser.add_argument('--output_file', type=str, help='the output file name')
    return parser.parse_args(argv)


def main():
    args = None
    args = parse_arguments(args)
    add_placeholder_on_ckpt(args.input_file, args.output_file)


if __name__ == '__main__':
    main()

