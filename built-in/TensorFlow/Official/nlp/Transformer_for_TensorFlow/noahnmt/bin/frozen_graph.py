# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
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

""" Generates model predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate
import sys
import os
import copy
import numpy as np

import tensorflow as tf
import yaml
from six import string_types
from tensorflow import gfile
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.core.protobuf import rewriter_config_pb2


from noahnmt.bin import train as nmt_train
from noahnmt.configurable import _maybe_load_yaml
from noahnmt.utils import cloud_utils
from noahnmt.utils import trainer_lib
from noahnmt.utils import decode_utils
# from noahnmt.utils import frozen_utils
from noahnmt.utils import train_utils
from noahnmt.utils import constant_utils
from noahnmt.utils import graph_utils
from noahnmt.utils import trainer_lib


flags = tf.flags
FLAGS = flags.FLAGS


tf.flags.DEFINE_string("output_filename", "inference.pb",
                       """output finalized graph filename""")
tf.flags.DEFINE_boolean("optimize", True, """optimize for inference, including:
                        - Stripping out parts of the graph that are never reached.
                        - fold_constants (not used because of error).""")
tf.flags.DEFINE_boolean("disable_vocab_table", False, """do not create vocab tables""")

FLAGS = tf.flags.FLAGS


def load_frozen_graph():
  with open(os.path.join(FLAGS.model_dir, FLAGS.output_filename), 'rb') as f:
    output_graph_def = tf.GraphDef()
    output_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(output_graph_def, name='')
      tf.logging.info("Import graph_def success.")
      with tf.Session(graph=graph, config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        # for op in graph.get_operations():
        #   tf.logging.info(op.name + " : " + op.device)
        # sess.run([])
        try:
          sess.run(graph.get_operation_by_name('init_all_tables'))
        except KeyError:
          pass
        
        if not FLAGS.disable_vocab_table:
          feed_dict={"source_tokens:0": [["1","1","1","1","1"]],
                     "source_len:0": [5]}
          ret = "predicted_tokens:0"
        else:
          feed_dict={"source_ids:0": [[0,0,0,0,0]],
                     "source_len:0": [5]}
          ret = "predicted_ids:0"
        
        sess.run(ret, feed_dict=feed_dict)



      #   try:
      #     sess.run(graph.get_operation_by_name('init_all_tables'))
      #   except KeyError:
      #     pass
      #   pred_tokens = sess.run("predicted_tokens:0", feed_dict={
      #       "source_tokens:0": [['1', '1', '1', '1', '1']],
      #       "source_len:0": [5]
      #       })
      #   if np.ndim(pred_tokens) > 2:
      #     print(pred_tokens[:,:,0])
      #   else:
      #     print(pred_tokens[:,:])


def main(_argv):
  """Program entry point.
  """
  # set log_file
  if FLAGS.log_file:
    cloud_utils.set_log_file(FLAGS.log_file)
  
  # There might be several model_dirs in ensemble decoding
  model_dirs = FLAGS.model_dir.strip().split(",")
  FLAGS.model_dir = model_dirs[0]

  # Load flags from config file
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  hparams = trainer_lib.create_hparams_from_flags(FLAGS)
  # hparams.init_from_scaffold = len(model_dirs)>1

  # update model name and model_params
  model_name, model_params = decode_utils.update_model_name_and_params(
      hparams, model_dirs=model_dirs)
  
  model_params["disable_vocab_table"] = FLAGS.disable_vocab_table

  # create estimator
  run_config = nmt_train.create_run_config(hparams)
  estimator = trainer_lib.create_estimator(
      model_name=model_name,
      model_params=model_params,
      run_config=run_config,
      hparams=hparams)

  # Build the graph
  if FLAGS.disable_vocab_table:
    features = {"source_ids": tf.placeholder(dtype=tf.int32, shape=[1, 128], name="source_ids"),
                "source_mask": tf.placeholder(dtype=constant_utils.DT_INT(),shape=[1, 128], name="source_mask"),
               }
  else:
    features = {"source_tokens": tf.placeholder(dtype=tf.string, shape=[None, None], name="source_tokens"),
                "source_len": tf.placeholder(dtype=constant_utils.DT_INT(),shape=[None], name="source_len"),
               }

  # create model and get predictions
  estimator_spec = estimator.model_fn(
      features=features, 
      labels=None, 
      mode=tf.estimator.ModeKeys.PREDICT,
      config=estimator.config)

  # checkpoint path
  checkpoint_path = FLAGS.checkpoint_path
  if not checkpoint_path:
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  

  # variables used when frozning graphs
  pred = {}
  if not FLAGS.disable_vocab_table:
    for key in ["predicted_tokens", "attention_scores"]:
      pred[key] = tf.identity(estimator_spec.predictions[key], name=key)
      output_node_names = [value.op.name for key, value in pred.items()] + ["init_all_tables"]
      input_node_names = ["source_tokens", "source_len"]
  else:
    for key in ["predicted_ids", "attention_scores"]:
      pred[key] = tf.identity(estimator_spec.predictions[key], name=key)

    output_node_names = [value.op.name for key, value in pred.items()]
    input_node_names = ["source_ids", "source_mask"]


  # saver = tf.train.Saver()
  # graph_options = tf.GraphOptions(
  #     optimizer_options=tf.OptimizerOptions(
  #         opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
  # graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
    
  # config = tf.ConfigProto(graph_options=graph_options)

  # with tf.Session(config=config) as sess:
  #   saver.restore(sess, checkpoint_path)
  #   tf.logging.info("Restored model from %s", checkpoint_path)
  #   sess.run(tf.tables_initializer())

  with tf.train.MonitoredSession(
          session_creator=tf.train.ChiefSessionCreator(
          checkpoint_filename_with_path=checkpoint_path,
          master=estimator.config.master,
          scaffold=estimator_spec.scaffold,
          config=estimator._session_config)) as mon_sess:
    
    # saver.save(sess._tf_sess, FLAGS.model_dir + "/inference.ckpt", global_step=0)
    # We use a built-in TF helper to export variables to constants
    tf.logging.info("Converting variables to constants...")
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        mon_sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
        output_node_names # The output node names are used to select the usefull nodes
    )

    # fix decoder/embedding_lookup colocate error
    for node in output_graph_def.node:
      if "_class" in node.attr:
        del node.attr["_class"]
        tf.logging.warning("Removing _calss attr of %s"%node.name)

  # optimize for inference
  # TODO fold_constants error
  if FLAGS.optimize:
    transforms = [
        "add_default_attributes",
        "remove_device",
        "strip_unused_nodes(type=float)",
        "merge_duplicate_nodes",
        "remove_nodes(op=CheckNumerics,op=StopGradient,op=Identity)",
        # "fold_constants(ignore_errors=true)",
        "sort_by_execution_order"]
    tf.logging.info("Optimizing for inference...")
    tf.logging.info("Before: %d ops in the graph." % len(output_graph_def.node))
    output_graph_def = TransformGraph(
        output_graph_def, 
        input_node_names, 
        output_node_names,
        transforms)
    tf.logging.info("After: %d ops in the graph." % len(output_graph_def.node))
    # remove Identity and CheckNumerics nodes
    tf.logging.info("Remove training nodes...")
    output_graph_def = tf.graph_util.remove_training_nodes(
        output_graph_def, protected_nodes=input_node_names+output_node_names)
    tf.logging.info("After: %d ops in the graph." % len(output_graph_def.node))

    # merge duplicate nodes
#     tf.logging.info("Merge duplicate nodes")
#     output_graph_def = opt4infer_utils.merge_duplicate_nodes(
#         output_graph_def, input_node_names, output_node_names)
#     tf.logging.info("After: %d ops in the graph." % len(output_graph_def.node))


  # Finally we serialize and dump the output graph to the filesystem
  with tf.gfile.GFile(os.path.join(FLAGS.model_dir, FLAGS.output_filename), "wb") as f:
      f.write(output_graph_def.SerializeToString())
  tf.logging.info("%d ops in the final graph." % len(output_graph_def.node))

  #load_frozen_graph()

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
