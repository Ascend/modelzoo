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
from model import yolov3
from utils.misc_utils import parse_anchors,read_class_names
import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default=1,
                        help="""set checkpoint path""")
    parser.add_argument("--anchor_path", type=str, default='./data/yolo_anchors.txt',
                        help="The path of the anchor txt file.")
    parser.add_argument("--class_name_path", type=str, default='./data/coco.names',
                        help="The path of the class names.")
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
    inputs = tf.placeholder(tf.float32, shape=[None, 416, 416, 3], name="input")
    #read classes and anchors
    classes = read_class_names(args.class_name_path)
    num_class = len(classes)
    anchors = parse_anchors(args.anchor_path)
    # create inference graph
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(inputs, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    #freeze graph
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './', 'model.pb')    #save unfrozen graph
        freeze_graph.freeze_graph(
		        input_graph='./model.pb',   # unfrozen graph
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=args.ckpt_path,
		        output_node_names="concat_9,mul_9",  # graph outputs node
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='./yolov3_tf.pb',   # output pb name
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()
