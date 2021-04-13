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
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
from alexnet import alexnet
from npu_bridge.estimator import npu_ops
import argparse
import os

CUR_PATH = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default="",
                        help="""set checkpoint path""")
    parser.add_argument('--output_path', default=CUR_PATH,
                        help="""set output path""")
    parser.add_argument('--num_classes', default=1000, type=int,
                        help="""number of classes for datasets """)
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def main():
    args = parse_args()
    print("ckpt_path:{}, output_path:{}, num_classes: {}".format(
        args.ckpt_path, args.output_path, args.num_classes))
    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # create inference graph
    top_layer = alexnet.inference(inputs, version="he_uniform",
                                  num_classes=args.num_classes, is_training=False)

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, args.output_path, 'model.pb')  # save unfrozen graph
        freeze_graph.freeze_graph(
            input_graph=os.path.join(args.output_path, 'model.pb'),
            input_saver='',
            input_binary=False,
            input_checkpoint=args.ckpt_path,
            output_node_names='dense_2/Relu',  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(args.output_path, 'alexnet_tf_910.pb'),  # graph outputs name
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()
