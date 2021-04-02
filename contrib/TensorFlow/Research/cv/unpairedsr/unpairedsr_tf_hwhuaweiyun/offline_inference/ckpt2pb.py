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

import argparse
import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import low_high_model


def main():
    """

    Convert TensorFlow checkpoints to pb file.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of mse loss')
    parser.add_argument('--beta', type=float, default=0.05, help='Weight of generator loss')
    parser.add_argument('--ckpt_dir', type=str, default='../output', help='Directory for storing model checkpoints')
    parser.add_argument('--pb_dir', type=str, default='./pb_model', help='Directory for storing pb files')
    args = parser.parse_args()

    # Set TensorFlow's log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)

    # Instantiate a model instance
    model = low_high_model.MODEL(args)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=sess_config)

    # Load model from the checkpoints
    var_list = tf.trainable_variables('generator')
    g_list = tf.global_variables()
    var_list += [g for g in g_list if 'moving_mean' in g.name] + [g for g in g_list if 'moving_variance' in g.name]
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
    ckpt_path = os.path.join(args.ckpt_dir, os.path.basename(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt_path)

    # Output node
    outputs = tf.identity(model.fake_hr, name='output')

    # Save pb files
    if not os.path.isdir(args.pb_dir):
        os.makedirs(args.pb_dir)
    tf.train.write_graph(sess.graph_def, args.pb_dir, 'tmp.pb', as_text=True)
    freeze_graph.freeze_graph(
        input_graph=os.path.join(args.pb_dir, 'tmp.pb'),
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_path,
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(args.pb_dir, 'unpairedsr.pb'),
        clear_devices=False,
        initializer_nodes='')
    os.remove(os.path.join(args.pb_dir, 'tmp.pb'))


if __name__ == '__main__':
    main()
