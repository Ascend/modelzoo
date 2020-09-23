# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import time

from npu_bridge.estimator.npu.npu_config import *
from npu_bridge.estimator.npu.npu_estimator import *
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_estimator import NPUEstimatorSpec

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
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

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval_all", True, "Whether to run eval on the dev set all.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_bool("do_train_and_eval", False, "train and eval")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 10,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool('npu_bert_debug', False, 'If True, dropout and shuffle is disabled.')

flags.DEFINE_bool('npu_bert_fused_gelu', True, 'Whether to use npu defined gelu op')

flags.DEFINE_bool('npu_bert_npu_dropout', True, 'Whether to use npu defined gelu op')

flags.DEFINE_integer("npu_bert_loss_scale", 0, "Whether to use loss scale, -1 is disable, 0 is dynamic loss scale, >=1 is static loss scale")

flags.DEFINE_integer('init_loss_scale_value', 2**32, 'Initial loss scale value for loss scale optimizer')

flags.DEFINE_bool('hcom_parallel', True, 'Whether to use parallel allreduce')


class _LogSessionRunHook(tf.train.SessionRunHook):
  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=['loss_scale:0'])

  def after_run(self, run_context, run_values):
    loss_scaler = run_values.results
    print("loss_scale:", loss_scaler, flush=True)

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
#      language = tokenization.convert_to_unicode(line[0])
#      if language != tokenization.convert_to_unicode(self.language):
#        continue
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
  def get_test_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
#      language = tokenization.convert_to_unicode(line[0])
#      if language != tokenization.convert_to_unicode(self.language):
#        continue
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class LcqmcProcessor(DataProcessor):
  """Processor for the LCQMC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
 #     if set_type == "test":
 #       label = "0"
 #     else:
      label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples



import random
class IntentProcessor(DataProcessor):
  """Processor for the Intent classification."""
  def __init__(self):
      self.labels = []
  def get_train_examples(self, data_dir):
      file_path = os.path.join(data_dir, 'train.tsv')  # cnews.train.txt
      with open(file_path, 'r', encoding="utf-8") as f:
          reader = f.readlines()
      random.seed(0)
      random.shuffle(reader)  #

      examples = []
      for index, line in enumerate(reader):
          guid = 'train-%d' % index
          split_line = line.strip().split("\t")
          text_a = tokenization.convert_to_unicode(split_line[1])
          text_b = None
          label = split_line[0]
          examples.append(InputExample(guid=guid, text_a=text_a,
                                       text_b=text_b, label=label))
          self.labels.append(label)
      return examples

  def get_dev_examples(self, data_dir):
      file_path = os.path.join(data_dir, 'dev.tsv')
      with open(file_path, 'r', encoding="utf-8") as f:
          reader = f.readlines()

      examples = []
      for index, line in enumerate(reader):
          guid = 'train-%d' % index
          split_line = line.strip().split("\t")
          text_a = tokenization.convert_to_unicode(split_line[1])
          text_b = None
          label = split_line[0]
          examples.append(InputExample(guid=guid, text_a=text_a,
                                       text_b=text_b, label=label))

      return examples


  def get_test_examples(self, data_dir):
      file_path = os.path.join(data_dir, 'test.tsv')
      with open(file_path, 'r', encoding="utf-8") as f:
          reader = f.readlines()

      examples = []
      for index, line in enumerate(reader):
          guid = 'train-%d' % index
          split_line = line.strip().split("\t")
          text_a = tokenization.convert_to_unicode(split_line[1])
          text_b = None
          label = split_line[0]
          examples.append(InputExample(guid=guid, text_a=text_a,
                                       text_b=text_b, label=label))

      return examples

  def get_labels(self):
    """See base class."""
    return set(self.labels)


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      if not FLAGS.npu_bert_debug:
        d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


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

  with tf.variable_scope("classifier"):
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


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      hooks=[]
      if FLAGS.npu_bert_loss_scale == 0:
        hooks.append(_LogSessionRunHook())
      output_spec = NPUEstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold_fn, training_hooks=hooks)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
      output_spec = NPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics,
          scaffold=scaffold_fn)
    else:
      output_spec = NPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder, batch_size):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      if not FLAGS.npu_bert_debug:
        d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info("***** Setting parameters *****")
  FLAGS.vocab_file = os.path.join(FLAGS.data_dir, 'nezha_large_vocab.txt')
  FLAGS.bert_config_file = os.path.join(FLAGS.data_dir, 'nezha_large_config.json')
  task_name = FLAGS.task_name
  print("task_name is :", task_name)
  if task_name=='chnsenti':
    print("In chnsenti setting")
    FLAGS.data_dir = os.path.join(FLAGS.data_dir,'chnsenti')
    FLAGS.output_dir = os.path.join(FLAGS.output_dir,'chnsenti_output')
    FLAGS.max_seq_length=256
    FLAGS.train_batch_size=32
    FLAGS.eval_batch_size=32
    FLAGS.num_train_epochs=10
    FLAGS.save_checkpoints_steps=100
  elif task_name=='xnli':
    print("In xnli setting")
    FLAGS.data_dir = os.path.join(FLAGS.data_dir,'xnli')
    FLAGS.output_dir = os.path.join(FLAGS.output_dir,'xnli_output')
    FLAGS.max_seq_length=128
    FLAGS.train_batch_size=64
    FLAGS.eval_batch_size=64
    FLAGS.num_train_epochs=3
    FLAGS.save_checkpoints_steps=100
  elif task_name=='lcqmc':
    print("In lcqmc setting")
    FLAGS.data_dir = os.path.join(FLAGS.data_dir,'lcqmc')
    FLAGS.output_dir = os.path.join(FLAGS.output_dir,'lcqmc_output')
    FLAGS.max_seq_length=128
    FLAGS.train_batch_size=64
    FLAGS.eval_batch_size=64
    FLAGS.num_train_epochs=5
    FLAGS.save_checkpoints_steps=100
  else:
    raise ValueError("Model zoo classifier test only consider intent, xnli and lcqmc")

  for name, value in FLAGS.__flags.items():
    print("TF flags:", name, "      ", FLAGS[name].value)

  processors = {
      "xnli": XnliProcessor,
      "chnsenti": IntentProcessor,
      "lcqmc": LcqmcProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()


  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = NPURunConfig(
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=0,
      save_summary_steps=0,
      session_config=tf.ConfigProto(),
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      log_step_count_steps=1,
      enable_data_pre_proc=True,
      iterations_per_loop=FLAGS.iterations_per_loop,
      hcom_parallel=FLAGS.hcom_parallel)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  label_list=[]
  #get the label-list by training data 
  #if FLAGS.do_train:
  if FLAGS.do_train or FLAGS.do_predict:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    label_list=processor.get_labels()
    print("label_list:", label_list)
    with open(os.path.join(FLAGS.output_dir, "label_list.txt"),'w') as wr:
      for label in label_list:
        wr.write(label)
        wr.write('\n')
    
    time.sleep(10)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    print("num_train_steps:================", num_train_steps)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  else:
    with open(os.path.join(FLAGS.output_dir, "label_list.txt"),'r') as fr:
      lines=fr.readlines()
      for line in lines:
        label_list.append(line.strip())      
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = NPUEstimator(
      model_fn=model_fn,
      config=run_config,)

  best_ckpt = ""
  best_item = ""
  if FLAGS.do_train:

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        batch_size=FLAGS.train_batch_size)
    if FLAGS.do_train_and_eval:
      train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
    else:
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval or FLAGS.do_train_and_eval or FLAGS.do_eval_all:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.

    eval_drop_remainder = True
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        batch_size=FLAGS.eval_batch_size)

    print("ckpt dir is: "+FLAGS.output_dir)
    if FLAGS.do_train_and_eval:
      eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=1*60, throttle_secs=60)#eval_steps
    elif FLAGS.do_eval_all:
      steps_and_files = []
      eval_all_list = []
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
          global_step = int(cur_filename.split("-")[-1])
          tf.logging.info("Add {} to eval list.".format(cur_filename))
          steps_and_files.append([global_step,cur_filename])
      steps_and_files = sorted(steps_and_files, key = lambda x:x[0])
      output_eval_all_file = os.path.join(FLAGS.output_dir, "eval_all_result.txt")
      output_eval_all_file_sorted = os.path.join(FLAGS.output_dir, "eval_all_result_sorted.txt")
      tf.logging.info("output_eval_all_file: "+output_eval_all_file)
      with tf.gfile.GFile(output_eval_all_file, "w") as writer:
        for global_step, filename in sorted(steps_and_files, key = lambda x:x[0]):
          time.sleep(15)
          result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)
          eval_all_list.append([result["eval_accuracy"],global_step,filename])
          writer.write("*****Eval all results %s *****\n" % (filename))
          for key in sorted(result.keys()):
            tf.logging.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
      eval_all_list = sorted(eval_all_list, key=lambda x:x[0], reverse = True)
      if(len(eval_all_list)!=0):
        best_item = str(eval_all_list[0])
        best_ckpt = str(eval_all_list[0][2])
      with open(output_eval_all_file_sorted, "w") as writer:
        for ele in eval_all_list:
          writer.write(str(ele))
          writer.write('\n')
      # then preidict from best ckpt
      predict_examples = processor.get_test_examples(FLAGS.data_dir)
      tf.logging.info('*****************label_list for prediction is %s'%str(label_list))
      time.sleep(10)
      num_actual_predict_examples = len(predict_examples)
      predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
      file_based_convert_examples_to_features(predict_examples, label_list,
                                              FLAGS.max_seq_length, tokenizer,
                                              predict_file)

      tf.logging.info("***** Running prediction*****")
      tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                      len(predict_examples), num_actual_predict_examples,
                      len(predict_examples) - num_actual_predict_examples)
      tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

      predict_drop_remainder = True
      predict_input_fn = file_based_input_fn_builder(
          input_file=predict_file,
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=predict_drop_remainder,
          batch_size=FLAGS.predict_batch_size)
      result = estimator.evaluate(input_fn=predict_input_fn,checkpoint_path=best_ckpt)#predict
      output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
      eval_re_path=os.path.join(FLAGS.output_dir, "eval")
      if not os.path.exists(eval_re_path):
          os.mkdir(eval_re_path)
      output_test_file = os.path.join(eval_re_path, "test_results.txt")
      with tf.gfile.GFile(output_test_file, "w") as writer:
        tf.logging.info("***** Test results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))
        tf.logging.info("***** Test Best results *****")
        tf.logging.info(best_item)
        writer.write(best_item)
    else:
      result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
      with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_train_and_eval:
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  if FLAGS.do_predict:

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    tf.logging.info('*****************label_list for prediction is %s'%str(label_list))
    time.sleep(10)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        batch_size=FLAGS.predict_batch_size)

    result = estimator.evaluate(input_fn=predict_input_fn,checkpoint_path=FLAGS.init_checkpoint)#predict
    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    eval_re_path=os.path.join(FLAGS.output_dir, "eval")
    if not os.path.exists(eval_re_path):
        os.mkdir(eval_re_path) 
    output_test_file = os.path.join(eval_re_path, "test_results.txt")
    with tf.gfile.GFile(output_test_file, "w") as writer:
      tf.logging.info("***** Test results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  tf.app.run()
