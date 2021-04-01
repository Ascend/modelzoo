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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import collections
import modeling
#import optimization
import tokenization
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
import sys, os
sys.path.append('..')
#from classifier_utils import *
from run_ner import MsraNERProcessor

flags = tf.flags
FLAGS = flags.FLAGS


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, num_labels, use_one_hot_embeddings):
  model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=use_one_hot_embeddings
  )

  output_layer = model.get_sequence_output()
  hidden_size = output_layer.shape[-1].value
  output_weight = tf.get_variable(
    "output_weights", [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02)
  )
  output_bias = tf.get_variable(
    "output_bias", [num_labels], initializer=tf.zeros_initializer()
  )
  with tf.variable_scope("loss"):
    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
    probabilities = tf.nn.softmax(logits, axis=-1)
    predict = tf.argmax(probabilities, axis=-1)
    return predict



def get_frozen_model(bert_config, num_labels, use_one_hot_embeddings):
  model_file = tf.train.latest_checkpoint(FLAGS.output_dir)
  print("model_file:", model_file, "output_dir:", FLAGS.output_dir)
  with tf.Graph().as_default(), tf.Session() as tf_sess:
    input_ids = tf.placeholder(tf.int64, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int64, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int64, [None, FLAGS.max_seq_length], name='segment_ids')
    predicts = create_model(bert_config, False, input_ids, input_mask, segment_ids, num_labels, use_one_hot_embeddings)
    saver = tf.train.Saver()
    print("restore;{}".format(model_file))
    saver.restore(tf_sess, model_file)
    tmp_g = tf_sess.graph.as_graph_def()
    print('freeze...')
    frozen_graph = tf.graph_util.convert_variables_to_constants(tf_sess,tmp_g, ['loss/ArgMax'])
    out_graph_path = os.path.join(FLAGS.output_dir, "bert_msraner.pb")
    with tf.io.gfile.GFile(out_graph_path, "wb") as f:
      f.write(frozen_graph.SerializeToString())
    print(f'pb file saved in {out_graph_path}')
    print("******msraner ckp transfer to pb sucess!*******")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  processors = {
      "msraner": MsraNERProcessor
  }
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  task_name = FLAGS.task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))
  processor = processors[task_name]()
  label_list = processor.get_labels()
  print('label_list', label_list, "nums_label:", len(label_list) + 1)

  get_frozen_model(bert_config, len(label_list) + 1, False)

if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
