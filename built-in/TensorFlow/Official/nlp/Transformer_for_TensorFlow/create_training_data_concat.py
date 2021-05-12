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
#
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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
#"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_boolean("clip_to_max_len", False, "clip sequences to xaximum sequence length.")

flags.DEFINE_integer("random_seed", 1, "Random seed for data generation.")

flags.DEFINE_integer("num_splits", 16, "number of output files")


EOS = "</s>"
SOS = "<s>"


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, source_eos_tokens, target_sos_tokens, target_eos_tokens):
    self.source_eos_tokens = source_eos_tokens
    self.target_sos_tokens = target_sos_tokens
    self.target_eos_tokens = target_eos_tokens
    self.length = [len(source_eos_tokens),
                   len(target_sos_tokens),
                   len(target_eos_tokens)]

  def __str__(self):
    s = ""
    s += "source eos tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.source_eos_tokens]))
    s += "target sos tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.target_sos_tokens]))
    s += "target eos tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.target_eos_tokens]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_file(writer, instance, tokenizer, max_seq_length):
  """Create TF example files from `TrainingInstance`s."""
  
  def _convert_ids_and_mask(input_tokens, is_sos):  
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_seq_length

    # create segment_ids and position_ids
    segment_ids = []
    position_ids = []
    segment = 0
    position = 0
    for i, tok in enumerate(input_tokens):
      # sos case
      if is_sos and tok == SOS and i > 0:
        segment += 1
        position = 0

      segment_ids.append(segment)
      position_ids.append(position)
      position += 1

      # eos case
      if not is_sos and tok == EOS:
        segment += 1
        position = 0

    # padding
    segment += 1
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(segment)
      position_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(position_ids) == max_seq_length
    

    return input_ids, input_mask, segment_ids, position_ids

  
  source_eos_ids, source_eos_mask, source_eos_segids, source_eos_posids = _convert_ids_and_mask(instance.source_eos_tokens, False)
  target_sos_ids, target_sos_mask, target_sos_segids, target_sos_posids = _convert_ids_and_mask(instance.target_sos_tokens, True)
  target_eos_ids, target_eos_mask, target_eos_segids, target_eos_posids = _convert_ids_and_mask(instance.target_eos_tokens, False)

  features = collections.OrderedDict()
  features["source_eos_ids"] = create_int_feature(source_eos_ids)
  features["source_eos_mask"] = create_int_feature(source_eos_mask)
  features["source_eos_segids"] = create_int_feature(source_eos_segids)
  features["source_eos_posids"] = create_int_feature(source_eos_posids)

  features["target_sos_ids"] = create_int_feature(target_sos_ids)
  features["target_sos_mask"] = create_int_feature(target_sos_mask)
  features["target_sos_segids"] = create_int_feature(target_sos_segids)
  features["target_sos_posids"] = create_int_feature(target_sos_posids)

  features["target_eos_ids"] = create_int_feature(target_eos_ids)
  features["target_eos_mask"] = create_int_feature(target_eos_mask)
  features["target_eos_segids"] = create_int_feature(target_eos_segids)
  features["target_eos_posids"] = create_int_feature(target_eos_posids)


  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  writer.write(tf_example.SerializeToString())

  return features



def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature



def create_training_instance(source_words, target_words, max_seq_length):
  """Creates `TrainingInstance`s for a single document."""

  if len(source_words) >= max_seq_length or len(target_words) >= max_seq_length:
    if FLAGS.clip_to_max_len:
      source_words = source_words[:min([len(source_words, max_seq_length-1)])]
      target_words = target_words[:min([len(target_words, max_seq_length-1)])]
    else:
      return None

  source_eos_tokens = source_words + [EOS]
  target_sos_tokens = [SOS] + target_words
  target_eos_tokens = target_words + [EOS]

  instance = TrainingInstance(
    source_eos_tokens=source_eos_tokens,
    target_sos_tokens=target_sos_tokens,
    target_eos_tokens=target_eos_tokens)

  return instance


def load_instances(input_files, tokenizer):
  total_read = 0
  all_instances = []
  for input_file in input_files:
    tf.logging.info("*** Reading from   %s ***", input_file)
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        
        total_read += 1
        if total_read % 100000 == 0:
          tf.logging.info("%d ..." % total_read)

        source_line, target_line = line.strip().split("\t")        
        source_tokens = tokenizer.tokenize(source_line)
        target_tokens = tokenizer.tokenize(target_line)

        if len(source_tokens) >= FLAGS.max_seq_length or len(target_tokens) >= FLAGS.max_seq_length:
          tf.logging.info("ignore long sentence!")
          continue

        instance = create_training_instance(source_tokens, target_tokens, FLAGS.max_seq_length)
        if instance is None:
          continue
        
        all_instances.append(instance)
  return all_instances


# concat two instance
def combine_instance(ins1, ins2):
  instance = TrainingInstance(
    source_eos_tokens=ins1.source_eos_tokens + ins2.source_eos_tokens,
    target_sos_tokens=ins1.target_sos_tokens + ins2.target_sos_tokens,
    target_eos_tokens=ins1.target_eos_tokens + ins2.target_eos_tokens)
  return instance


def concat_sents(instances):
  max_skip = 1000
  visited = {}
  max_length = FLAGS.max_seq_length
  ret = []
  for i, instance in enumerate(instances):
    # skip visited instance
    if i in visited:
      continue
    visited[i] = 1

    # find another instance to concat
    skip_count = 0
    for j in range(i+1, len(instances)):
      if j in visited:
        continue
      # update skip_count
      skip_count += 1
      if skip_count > max_skip:
        break

      length = [x+y for x,y in zip(instance.length, instances[j].length)]
      if max(length) > max_length:
        continue
      else:
        instance = combine_instance(instance, instances[j])
        visited[j] = 1
    
    # now instance is already a concatenation of one or more instances
    ret.append(instance)
  return ret


  


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.WhiteSpaceTokenizer(
      vocab_file=FLAGS.vocab_file)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)
  
  output_files = []
  tf.logging.info("*** Writing to output files ***")
  if FLAGS.num_splits > 1:
    for i in range(FLAGS.num_splits):
      output_file = FLAGS.output_file + "-%03d-of-%03d" % (i+1, FLAGS.num_splits)
      output_files.append(output_file)
      tf.logging.info("  %s", output_file)
  else:
    output_file = FLAGS.output_file
    output_files.append(output_file)
    tf.logging.info("  %s", output_file)

  
  # output_files = FLAGS.output_file.split(",")
  # tf.logging.info("*** Writing to output files ***")
  # for output_file in output_files:
  #   tf.logging.info("  %s", output_file)

  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  random.seed(FLAGS.random_seed)
  # rng = random.Random(FLAGS.random_seed)

  # load all lines
  tf.logging.info("load all lines ...")
  instances= load_instances(input_files, tokenizer)

  # shuffle
  tf.logging.info("shuffle ...")
  random.shuffle(instances)

  # concat short sentences to max_length
  tf.logging.info("concat short sentences ...")
  instances = concat_sents(instances)

  #
  tf.logging.info("In total %d instances" % len(instances))

  tf.logging.info("writing to file ...")
  # write to file
  writer_index = 0
  total_written = 0
  for instance in instances:
    features = write_instance_to_file(writers[writer_index], instance, tokenizer, FLAGS.max_seq_length)

    writer_index = (writer_index + 1) % len(writers)
    total_written += 1

    if total_written <= 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("source tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.source_eos_tokens]))
      tf.logging.info("target tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.target_sos_tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  
  for writer in writers:
        writer.close()

  tf.logging.info("Wrote %d total instances", total_written)



if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
