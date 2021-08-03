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
from tensorflow.python.framework import graph_util
import argparse

parser = argparse.ArgumentParser(description="solidified PB model.")
parser.add_argument("--input_ckpt", type=str, default="./model.ckpt", help="input checkpoint.")
parser.add_argument("--output_pb", type=str, default="./node.pb", help="output pb model.")
parser.add_argument("--output_node_name", type=str, default="",help="Output node names. Use commas(,) to separate multiple nodes.")
args= parser.parse_args()

def freeze_graph(input_checkpoint, output_graph, out_node_name):
    '''
    :param input_checkpoint:
    :param output_graph
    :param out_node_name:
    :return:
    '''
    input_node = tf.placeholder(tf.float32, [None, 640, 640, 3], name='input')
    out_node_names = out_node_name
    saver = tf.train.import_meta_graph(input_checkpoint+".meta", clear_devices=True)
    graph = tf.get_default_graph()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, input_checkpoint)
        input_graph_def = graph.as_graph_def()
        
        del input_graph_def.node[1:69]
        for node in input_graph_def.node:
            if node.name=="conv1_1_3x3_s2/Conv2D":
                node.input[0] = "input"
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess, input_graph_def=input_graph_def, output_node_names=out_node_names.split(","))
        with tf.gfile.GFile(output_graph,"wb") as f:
            f.write(output_graph_def.SerailizeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    input_checkpoint = args.input_ckpt
    output_graph = args.output_pb
    out_node_name = args.output_node_name
    freeze_graph(input_checkpoint, output_graph, out_node_name)
