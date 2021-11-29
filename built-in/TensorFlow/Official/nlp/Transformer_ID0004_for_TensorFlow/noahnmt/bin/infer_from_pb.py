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
import time
import codecs
import copy
import os

import tensorflow as tf
import numpy as np
import yaml
from six import string_types
from tensorflow import gfile
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.profiler import model_analyzer

from noahnmt.configurable import _maybe_load_yaml, _deep_merge_dict
from noahnmt.utils import device_utils
from noahnmt.utils import profile_utils


tf.flags.DEFINE_string("model", None,
                       """relative path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")
tf.flags.DEFINE_string("model_dir", "", "directory to load model from")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")
tf.flags.DEFINE_string("input", None, "input file")
tf.flags.DEFINE_string("output", None, "output file")
tf.flags.DEFINE_string("decode_params", "", "DecodeText params, YAML format")
tf.flags.DEFINE_integer("num_intra_threads", 0,
                       "Number of threads to use for intra-op parallelism. "
                       "If set to 0, the system will pick an appropriate number.")
tf.flags.DEFINE_integer("num_inter_threads", 0,
                       "Number of threads to use for inter-op parallelism. "
                       "If set to 0, the system will pick an appropriate number.")
tf.flags.DEFINE_boolean("enable_graph_rewriter", True, "enable graph rewriter to optimize graph (experimental)")
tf.flags.DEFINE_integer("profile_start_step", 0, "capture metadata from this step, default(0=disabled)")
tf.flags.DEFINE_integer("profile_stop_step", 0, "capture metadata to this step, default(0=disabled)")
tf.flags.DEFINE_integer("profile_every_n_steps", 1, "capture metadata every this steps, default(0=disabled)")
tf.flags.DEFINE_string("profile_views", "op", "list of views of output: graph, op, scope, code")
tf.flags.DEFINE_boolean("log_device_placement", False, "log device placement")
tf.flags.DEFINE_boolean("enable_jit", False, "enable XLA default False")


FLAGS = tf.flags.FLAGS


def _parse_params(params, default_params):
  """Parses parameter values to the types defined by the default parameters.
  Default parameters are used for missing values.
  """
  # Cast parameters to correct types
  if params is None:
    params = {}
  result = copy.deepcopy(default_params)
  for key, value in params.items():
    # If param is unknown, drop it to stay compatible with past versions
    if key not in default_params:
      raise ValueError("%s is not a valid model parameter" % key)
    # Param is a dictionary
    if isinstance(value, dict):
      default_dict = default_params[key]
      if not isinstance(default_dict, dict):
        raise ValueError("%s should not be a dictionary", key)
      if default_dict:
        value = _parse_params(value, default_dict)
      else:
        # If the default is an empty dict we do not typecheck it
        # and assume it's done downstream
        pass
    if value is None:
      continue
    if default_params[key] is None:
      result[key] = value
    else:
      result[key] = type(default_params[key])(value)
  return result


def unbatch_dict(dict_):
  """Converts a dictionary of batch items to a batch/list of
  dictionary items.
  """
  batch_size = list(dict_.values())[0].shape[0]
  for i in range(batch_size):
    yield {key: value[i] for key, value in dict_.items()}


def _get_prediction_length(predictions_dict):
  """Returns the length of the prediction based on the index
  of the first SEQUENCE_END token.
  """
  tokens_iter = enumerate(predictions_dict["predicted_tokens"])
  return next(((i + 1) for i, _ in tokens_iter if _ == "</s>"),
              len(predictions_dict["predicted_tokens"]))


def _get_unk_mapping(filename):
  """Reads a file that specifies a mapping from source to target tokens.
  The file must contain lines of the form <source>\t<target>"

  Args:
    filename: path to the mapping file

  Returns:
    A dictionary that maps from source -> target tokens.
  """
  with tf.gfile.GFile(filename, "r") as mapping_file:
    lines = mapping_file.readlines()
    mapping = dict([_.split("\t")[0:2] for _ in lines])
    mapping = {k.strip(): v.strip() for k, v in mapping.items()}
  return mapping


def _unk_replace(source_tokens,
                 predicted_tokens,
                 attention_scores,
                 mapping=None):
  """Replaces UNK tokens with tokens from the source or a
  provided mapping based on the attention scores.

  Args:
    source_tokens: A numpy array of strings.
    predicted_tokens: A numpy array of strings.
    attention_scores: A numeric numpy array
      of shape `[prediction_length, source_length]` that contains
      the attention scores.
    mapping: If not provided, an UNK token is replaced with the
      source token that has the highest attention score. If provided
      the token is insead replaced with `mapping[chosen_source_token]`.

  Returns:
    A new `predicted_tokens` array.
  """
  result = []
  for token, scores in zip(predicted_tokens, attention_scores):
    if token == "<unk>":
      max_score_index = np.argmax(scores)
      chosen_source_token = source_tokens[max_score_index]
      new_target = chosen_source_token
      if mapping is not None and chosen_source_token in mapping:
        new_target = mapping[chosen_source_token]
      result.append(new_target)
    else:
      result.append(token)
  return np.array(result)


def strip_bpe(text):
  """Deodes text that was processed using BPE from
  https://github.com/rsennrich/subword-nmt"""
  return text.replace("@@ ", "").strip()


def decode_sentencepiece(text):
  """Decodes text that uses https://github.com/google/sentencepiece encoding.
  Assumes that pieces are separated by a space"""
  return "".join(text.split(" ")).replace("▁", " ").strip()


class DecodeText():
  """Defines inference for tasks where both the input and output sequences
  are plain text.

  Params:
    delimiter: Character by which tokens are delimited. Defaults to space.
    unk_replace: If true, enable unknown token replacement based on attention
      scores.
    unk_mapping: If `unk_replace` is true, this can be the path to a file
      defining a dictionary to improve UNK token replacement. Refer to the
      documentation for more details.
    dump_attention_dir: Save attention scores and plots to this directory.
    dump_attention_no_plot: If true, only save attention scores, not
      attention plots.
    dump_beams: Write beam search debugging information to this file.
  """
  def __init__(self, params):
    self._params = _parse_params(params, self.default_params())
    self._unk_mapping = None
    self._unk_replace_fn = None

    if self.params["unk_mapping"] is not None:
      self._unk_mapping = _get_unk_mapping(self.params["unk_mapping"])
    if self.params["unk_replace"]:
      self._unk_replace_fn = functools.partial(
          _unk_replace, mapping=self._unk_mapping)

    self._postproc_fn = None
    if self.params["postproc_fn"]:
      self._postproc_fn = eval(self.params["postproc_fn"])
      if self._postproc_fn is None:
        raise ValueError("postproc_fn not found: {}".format(
            self.params["postproc_fn"]))

    # if self.params["output"] is None:
    #   raise ValueError("Please provide a filename to store translations")

  def default_params(self):
    params = {}
    params.update({
        "delimiter": " ",
        "postproc_fn": "",
        "unk_replace": False,
        "unk_mapping": None,
        "output": None
    })
    return params

  @property
  def params(self):
    return self._params


  def __call__(self, predictions):
    fetches_batch = predictions
    translations = []
    for fetches in unbatch_dict(fetches_batch):
      # Convert to unicode
      fetches["predicted_tokens"] = np.char.decode(
          fetches["predicted_tokens"].astype("S"), "utf-8")
      predicted_tokens = fetches["predicted_tokens"]

      # If we're using beam search we take the first beam
      beam_search = False
      if np.ndim(predicted_tokens) > 1:
        predicted_tokens = predicted_tokens[:, 0]
        beam_search = True

      if self._unk_replace_fn is not None:
        # We slice the attention scores so that we do not
        # accidentially replace UNK with a SEQUENCE_END token
        fetches["source_tokens"] = np.char.decode(
          fetches["source_tokens"].astype("S"), "utf-8")
        source_tokens = fetches["source_tokens"]
        source_len = fetches["source_len"]
        attention_scores = fetches["attention_scores"]
        if beam_search:
          attention_scores = attention_scores[:, 0]
        attention_scores = attention_scores[:, :source_len]
        predicted_tokens = self._unk_replace_fn(
            source_tokens=source_tokens,
            predicted_tokens=predicted_tokens,
            attention_scores=attention_scores)

      sent = self.params["delimiter"].join(predicted_tokens).split(
          "</s>")[0]
      sent = sent.split("<s>")[-1]

      # Apply postproc
      if self._postproc_fn:
        sent = self._postproc_fn(sent)

      sent = sent.strip()
      translations.append(sent)
    return translations


def input_generator(input_path, batch_size):

  def process_batch(batch):
    length = [len(x) for x in batch]
    maxlen = max(length)
    batch = [x + ["</s>"]*(maxlen-len(x)) if len(x) < maxlen else x for x in batch]
    return {"source_tokens:0": batch, 
            "source_len:0": length}

  f = tf.gfile.GFile(input_path, 'r')
  batch = []
  batch_size = max(1, batch_size)
  for line in f:
    tokens = line.strip().split()
    batch.append(tokens + ["</s>"])
    if len(batch) == batch_size:
      yield process_batch(batch)
      batch = []
  
  if len(batch) > 0:
    yield process_batch(batch)



def main(_argv):
  """Program entry point.
  """
  decode_text = DecodeText(
      _maybe_load_yaml(FLAGS.decode_params))
  inputs = input_generator(FLAGS.input, FLAGS.batch_size)
  output_file = tf.gfile.GFile(FLAGS.output, 'w')

  hooks = []

  graph_options = tf.GraphOptions(
      optimizer_options=tf.OptimizerOptions(
          opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
  if FLAGS.enable_graph_rewriter:
    graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
  if FLAGS.enable_jit:
    graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=graph_options,
      log_device_placement=FLAGS.log_device_placement,
      inter_op_parallelism_threads = FLAGS.num_inter_threads,
      intra_op_parallelism_threads = FLAGS.num_intra_threads)

  device = "cpu"
  if device_utils.get_num_gpus() > 0:
    device = "gpu"

  FLAGS.model = os.path.join(FLAGS.model_dir, FLAGS.model)

  total_time = 0
  with open(os.path.join(FLAGS.model), 'rb') as f:
    output_graph_def = tf.GraphDef()
    output_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph, tf.device("/%s:0" % device):
      tf.import_graph_def(output_graph_def, name='')
      tf.logging.info("Import graph_def success.")
      
      session_creator = tf.train.ChiefSessionCreator(
          config=sess_config)
      with tf.train.MonitoredSession(
          session_creator=session_creator,
          hooks=hooks) as sess:

        try:
          sess.run(graph.get_operation_by_name('init_all_tables'))
        except KeyError:
          pass

        # warm up
        for i in range(5):
          _ = sess.run("predicted_tokens:0", 
                       feed_dict={
                          "source_tokens:0": [["this", "is", "warm", "up", "</s>"]], 
                          "source_len:0": [5]})

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        nmt_profiler = model_analyzer.Profiler(graph=graph)
        out_flag = False

        counter = 0
        for batch in inputs:
          counter += 1

          step_run_options = None
          step_run_metadata = None
          if counter >= FLAGS.profile_start_step \
             and counter <= FLAGS.profile_stop_step \
             and (counter-FLAGS.profile_start_step) % FLAGS.profile_every_n_steps == 0:
            step_run_options = run_options
            step_run_metadata = run_metadata
            out_flag = True
            tf.logging.info("Collect run_metadata at step %d" % counter)          

          # run a batch and accumulate time
          start_time = time.time()
          pred_tokens = sess.run(
              "predicted_tokens:0", 
              feed_dict=batch, 
              options=step_run_options, 
              run_metadata=step_run_metadata)
          total_time += time.time() - start_time
          # post processing and output
          translations = decode_text({"predicted_tokens": pred_tokens})
          output_file.write("\n".join(translations) + "\n")

          if step_run_metadata is not None:
            nmt_profiler.add_step(step=counter, run_meta=step_run_metadata)
            if counter >= FLAGS.profile_stop_step:
              tf.logging.info("Break inference because of profiling!")
              break
        
        if out_flag:
          profile_utils.write_profiler(
              profiler=nmt_profiler, 
              views=FLAGS.profile_views.split(","), 
              model_dir=FLAGS.model_dir)
          profile_utils.write_metadata(
              run_metadata=run_metadata, 
              model_dir=FLAGS.model_dir, 
              step_done=counter)

  output_file.close()
  print("Total time: %s" % total_time, flush=True)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
