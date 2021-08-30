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
import numpy as np
import sys
sys.path.append('..')
from classifier_utils import *

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def get_frozen_model(bert_config, num_labels, use_one_hot_embeddings):
  tf_config = tf.compat.v1.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  output_node_names = ['loss/BiasAdd']

  model_file = tf.train.latest_checkpoint(FLAGS.output_dir)
  print("==========model_file:", model_file, "output_dir:", FLAGS.output_dir)
  with tf.Graph().as_default(), tf.Session(config=tf_config) as tf_sess:
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')

    create_model(bert_config, False, input_ids, input_mask, segment_ids, label_ids,
             num_labels, use_one_hot_embeddings)
    saver = tf.train.Saver()
    print("restore;{}".format(model_file))
    saver.restore(tf_sess, model_file)
    tmp_g = tf_sess.graph.as_graph_def()
    print('freeze...')
    frozen_graph = tf.graph_util.convert_variables_to_constants(tf_sess,
                                                            tmp_g, output_node_names)
    out_graph_path = os.path.join(FLAGS.output_dir, "Bert_tnews.pb")
    with tf.io.gfile.GFile(out_graph_path, "wb") as f:
      f.write(frozen_graph.SerializeToString())
    print(f'pb file saved in {out_graph_path}')
    print("******ckp transfer to pb sucess!")

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "xnli": XnliProcessor,
      "tnews": TnewsProcessor,
      "afqmc": AFQMCProcessor,
      "iflytek": iFLYTEKDataProcessor,
      "copa": COPAProcessor,
      "cmnli": CMNLIProcessor,
      "wsc": WSCProcessor,
      "csl": CslProcessor,
      "copa": COPAProcessor,
  }

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()
  print('label_list', label_list, "nums_label:", len(label_list))

  get_frozen_model(bert_config, len(label_list), False)

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
