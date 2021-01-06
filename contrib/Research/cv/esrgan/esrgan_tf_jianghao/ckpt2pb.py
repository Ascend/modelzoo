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

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from train_module import Network
import os


def set_flags():
    Flags = tf.app.flags

    Flags.DEFINE_string('checkpoint_dir', './ascend', 'checkpoint directory')
    Flags.DEFINE_string(
        'inference_checkpoint', 'pre_gen-480000',
        'checkpoint to use for inference. Empty string means the latest checkpoint is used'
    )

    Flags.DEFINE_integer('channel', 3, 'Number of input/output image channel')
    Flags.DEFINE_integer('num_repeat_RRDB', 23,
                         'The number of repeats of RRDB blocks')
    Flags.DEFINE_float('residual_scaling', 0.2, 'residual scaling parameter')
    Flags.DEFINE_integer('initialization_random_seed', 111,
                         'random seed of networks initialization')

    return Flags.FLAGS


def main():
    # set flag
    FLAGS = set_flags()
    tf.reset_default_graph()
    # build Generator
    input_data = tf.placeholder(tf.float32,
                                shape=[1, None, None, 3],
                                name='LR_input')
    network = Network(FLAGS, input_data)
    gen_out = network.generator()
    # start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ckpt_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.inference_checkpoint)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(var_list=tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
        saver.restore(sess, ckpt_path)

        tf.train.write_graph(sess.graph_def,
                             ckpt_path,
                             './ESRGAN.pb',
                             as_text=True)

        freeze_graph.freeze_graph(
            input_graph='./ESRGAN.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='generator/generator/last_conv/conv_2/BiasAdd',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./ESRGAN_frozen.pb',
            clear_devices=False,
            initializer_nodes='')

    print('done')


if __name__ == '__main__':
    main()
