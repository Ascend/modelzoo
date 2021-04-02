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
# coding:utf-8
import tensorflow as tf
import sys
from tensorflow.python.framework import graph_util
from npu_bridge.estimator import npu_ops
from resnet_model import Model
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
    with tf.Session() as sess:
        #modify input modes
        input_data = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_data')
        #create unfrozen graph
        model = Model(resnet_size=50, bottleneck=True, num_classes=1001, num_filters=64,
                   kernel_size=7,
                   conv_stride=2, first_pool_size=3, first_pool_stride=2,
                   block_sizes=[3, 4, 6, 3], block_strides=[1, 2, 2, 2],
                   resnet_version=1,)
        logits = model(input_data, False)
        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt_path)   #from xxxx.ckpt-xxxx.data load graph
        #convert variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=['resnet_model/final_dense'])
        with tf.gfile.GFile("resnet50_npu.pb", "wb") as f:  #  save GraphDef to pb
            f.write(output_graph_def.SerializeToString())
        print("Success %d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    main()
