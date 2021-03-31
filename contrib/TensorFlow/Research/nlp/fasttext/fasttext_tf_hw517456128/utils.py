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

import os
import hashlib
from shutil import copy

import numpy as np
import tensorflow as tf
from tensorflow.python import graph_util


def validate(path):
    if '"' in path:
        path = path.split('"')[1]
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def freeze_save_graph(sess, log_dir, name, output_node):
    for node in sess.graph.as_graph_def().node:
        node.device = ""

    variable_graph_def = sess.graph.as_graph_def()
    optimized_net = graph_util.convert_variables_to_constants(sess, variable_graph_def, [output_node])
    tf.io.write_graph(optimized_net, log_dir, name, False)


def load_graph(graph_path, return_elements=None):
    with tf.gfile.GFile(graph_path, 'rb') as infile:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(infile.read())
        output_nodes = tf.import_graph_def(graph_def, return_elements=return_elements, name="")
        return output_nodes


def hash_(value):
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def hash2_(value):
    return int.from_bytes(hashlib.md5(str(value).encode("utf-8")).digest(), "little")


def hash_xor_data(list_of_texts):
    init_hash = hash2_(list_of_texts[0])
    for text in list_of_texts[1:]:
        init_hash = init_hash ^ hash2_(text)
    return str(init_hash)


def get_cache_hash(list_of_texts, data_specific_params):
    data_hashed = "".join([str(v) for k, v in sorted(data_specific_params.items(), key=lambda t: t[0])])
    hash_xor = hash_xor_data(list_of_texts=list_of_texts)
    return hash_(data_hashed + hash_xor)


def handle_space_paths(path):
    if " " in path:
        return '"{}"'.format(path)
    return path


def copy_all(list_of_paths, destination_path):
    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)
    for src_path in list_of_paths:
        if os.path.isfile(src_path):
            copy(src_path, os.path.join(destination_path, os.path.basename(src_path)))
        else:
            print("Invalid path, no such file {}".format(src_path))


def percent_array(x, multiplier=100, precision=2):
    return np.round(multiplier * np.mean(x), precision)


def write_summaries(end_epoch_loss, mean_accuracy, mean_accuracy_k, top_k, summary_writer, epoch):
    summary_accuracy = tf.Summary(value=[tf.Summary.Value(tag="Accuracy",
                                                          simple_value=mean_accuracy)])
    summary_accuracy_k = tf.Summary(value=[tf.Summary.Value(tag="Accuracy_top_{}".format(top_k),
                                                            simple_value=mean_accuracy_k)])
    summary_loss = tf.Summary(
        value=[tf.Summary.Value(tag="Loss", simple_value=np.mean(end_epoch_loss))])

    summary_writer.add_summary(summary_accuracy, epoch)
    summary_writer.add_summary(summary_accuracy_k, epoch)
    summary_writer.add_summary(summary_loss, epoch)
