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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
from six.moves import xrange


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","))
    output_graph_def = graph_util.remove_training_nodes(output_graph_def,
                                                        protected_nodes=None)
    return output_graph_def


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ckpt_file= args.get('meta_file')

            saver = tf.train.import_meta_graph(ckpt_file+'.meta',
                                               clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), ckpt_file)

            # Retrieve the protobuf graph definition
            # and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()

            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def,
                                                args.get('output_nodes'))

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.get('output_file'), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node),
                                                 args.get('output_file')))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', type=str,
                        help='the metagraph (.meta) file')
    parser.add_argument('--output_file', type=str,
                        help='Filename for the exported graphdef protobuf')
    parser.add_argument('--output_nodes', type=str,
                        help='example:  tensorname1,tensorname2')
    return parser.parse_args(argv)


if __name__ == '__main__':
    # args = ['/job/output/logs/ckpt_first/placeholder/model.ckpt-1000',
    #         '/job/output/model/resnet_placeholder.pb',
    #         'fp32_vars/final_dense']
    args = None
    parsed_args = parse_arguments(args)
    model_input = {'meta_file': parsed_args.meta_file,
                   'output_file': parsed_args.output_file,
                   'output_nodes': parsed_args.output_nodes}
    main(model_input)
    # main(parse_arguments(sys.argv[1:]))
