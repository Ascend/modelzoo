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
from utils.misc_utils import parse_anchors, read_class_names


def freeze_graph_def(**args):
    """
    frozen the model of ckpt
    params:
        args: type is dict, include class_name_path, anchor_path, ckpt_path
    """
    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, 416, 416, 3], name="input")
    # read classes and anchors
    classes = read_class_names(args.get('class_name_path'))
    num_class = len(classes)
    anchors = parse_anchors(args.get('anchor_path'))
    # create inference graph
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(inputs, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    # freeze graph
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, '/cache/training', 'model.pb')    # save unfrozen graph
        freeze_graph.freeze_graph(
            input_graph='/cache/training/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=args.get('ckpt_path'),
            output_node_names="concat_9,mul_9",
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='/cache/training/yolov3_tf.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")


def transform_images_path(input_path, output_path):
    """
    transform the images path to suit the modelarts
    params:
        input_path: the original txt file
        output_path: after transform txt file
    """
    output_list = []
    with open(input_path, 'r') as f:
        ban_list = f.read().split('\n')[:-1]
        for item in ban_list:
            item_list = item.split(' ')
            image_name = item_list[1].split('/')[-1]
            dir_name = item_list[1].split('/')[-2]
            revert_path = '/cache/data_url/{}/{}'.format(dir_name, image_name)
            item_list[1] = revert_path
            output_list.append(item_list)

    f = open(output_path, 'w')
    for value in output_list:
        f.write(" ".join(value))
        f.write("\n")
    f.close()
