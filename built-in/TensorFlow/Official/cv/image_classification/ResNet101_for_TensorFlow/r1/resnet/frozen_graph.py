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
from official.r1.resnet import resnet_model
from tensorflow.contrib import slim
from npu_bridge.estimator import npu_ops
import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default=1,
                        help="""set checkpoint path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def main():
    args = parse_args()
    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # create inference graph
    model = resnet_model.Model(resnet_size=101, bottleneck=True, num_classes=1001, num_filters=64,
                   kernel_size=7,
                   conv_stride=2, first_pool_size=3, first_pool_stride=2,
                   block_sizes=[3, 4, 23, 3], block_strides=[1, 2, 2, 2],
                   resnet_version=1,)

    logits = model(inputs, False)

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './', 'model.pb')
        freeze_graph.freeze_graph(
		        input_graph='./model.pb',
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=args.ckpt_path,
		        output_node_names='resnet_model/final_dense',  #graph outputs node
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='./resnet101_tf_910.pb',   # graph outputs name
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()
