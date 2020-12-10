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
from inception import inception_v4
from tensorflow.contrib import slim as slim
from npu_bridge.estimator import npu_ops
import argparse

#set checkpoint path
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default="",
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
    # modify input node
    inputs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="input")
    # build inference graph
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        top_layer, end_points = inception_v4.inception_v4(inputs=inputs, num_classes=1000, dropout_keep_prob=1.0, is_training = False)
    logits = top_layer
    logits = tf.cast(logits, tf.float32)

    with tf.Session() as sess:
        #save unfrozen graph
        tf.train.write_graph(sess.graph_def, './', 'model.pb')    # save unfrozen graph
        #start to froze graph
        freeze_graph.freeze_graph(
		        input_graph='./model.pb',   # unfrozen graph
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=args.ckpt_path,
		        output_node_names='InceptionV4/Logits/Logits/BiasAdd',  # output node
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='inception_v4_tf.pb',   #saved pb name
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()
