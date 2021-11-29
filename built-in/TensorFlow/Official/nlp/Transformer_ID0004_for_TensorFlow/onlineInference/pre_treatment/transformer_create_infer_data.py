# transformer online inference data create
# -*- coding: utf-8 -*-
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import random
import tokenization
import tensorflow as tf
import numpy as np
import shutil


parser = argparse.ArgumentParser(description='Create input data of inference')
parser.add_argument('--input_file', type=str, default='./test.all', help='input file for create inference input')
parser.add_argument('--output_file', type=str,  default='./newstest2014-l128-mindrecord', help='infer result folder')
parser.add_argument('--vocab_file', type=str, default='../datasets/vocab.share', help='vocable file')

parser.add_argument('--max_seq_length', type=int, default=128, help='sequence max length')
parser.add_argument('--clip_to_max_len', type=bool, default=False, help='whether False or True')
parser.add_argument('--num_splits', type=int, default=16, help='split number')
parser.add_argument('--save_dir', type=str, default='../datasets/transformer', help='pre process out file dir')


class TrainingInstance(object):


    """A single training instance (sentence pair)."""

    def __init__(self, source_sos_tokens, source_eos_tokens, target_sos_tokens, target_eos_tokens):
        self.source_sos_tokens = source_sos_tokens
        self.source_eos_tokens = source_eos_tokens
        self.target_sos_tokens = target_sos_tokens
        self.target_eos_tokens = target_eos_tokens

    def __str__(self):
        s = ""
        s += "source sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_sos_tokens]))
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
    
    def _convert_ids_and_mask(input_tokens):  
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

  
    source_sos_ids, source_sos_mask = _convert_ids_and_mask(instance.source_sos_tokens)
    source_eos_ids, source_eos_mask = _convert_ids_and_mask(instance.source_eos_tokens)

    target_sos_ids, target_sos_mask = _convert_ids_and_mask(instance.target_sos_tokens)
    target_eos_ids, target_eos_mask = _convert_ids_and_mask(instance.target_eos_tokens)

    features = collections.OrderedDict()
    features["source_sos_ids"] = create_int_feature(source_sos_ids)
    features["source_sos_mask"] = create_int_feature(source_sos_mask)
    features["source_eos_ids"] = create_int_feature(source_eos_ids)
    features["source_eos_mask"] = create_int_feature(source_eos_mask)
    features["target_sos_ids"] = create_int_feature(target_sos_ids)
    features["target_sos_mask"] = create_int_feature(target_sos_mask)
    features["target_eos_ids"] = create_int_feature(target_eos_ids)
    features["target_eos_mask"] = create_int_feature(target_eos_mask)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

    return features


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature



def create_training_instance(source_words, target_words, max_seq_length, clip_to_max_len):
    """Creates `TrainingInstance`s for a single document."""
    EOS = "</s>"
    SOS = "<s>"

    if len(source_words) >= max_seq_length or len(target_words) >= max_seq_length:
        if clip_to_max_len:
            source_words = source_words[:min([len(source_words, max_seq_length-1)])]
            target_words = target_words[:min([len(target_words, max_seq_length-1)])]
        else:
            return None

    source_sos_tokens = [SOS] + source_words
    source_eos_tokens = source_words + [EOS]
    target_sos_tokens = [SOS] + target_words
    target_eos_tokens = target_words + [EOS]

    instance = TrainingInstance(
      source_sos_tokens=source_sos_tokens,
      source_eos_tokens=source_eos_tokens,
      target_sos_tokens=target_sos_tokens,
      target_eos_tokens=target_eos_tokens)

    return instance



def create_infer_input_data(inputFile, vocabFile, outputFile, max_seq_length, clip_to_max_len, num_splits, save_dir):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.WhiteSpaceTokenizer(
        vocab_file=vocabFile)

    input_files = []
    input_files.append(inputFile)
    #for input_pattern in inputFile.split(","):
    #    input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)
    
    output_files = []
    tf.logging.info("*** Writing to output files ***")
    if num_splits > 1:
        for i in range(num_splits):
            output_file = outputFile + "-%03d-of-%03d" % (i+1, num_splits)
            output_files.append(output_file)
            tf.logging.info("  %s", output_file)
    else:
        output_file = outputFile
        output_files.append(output_file)
        tf.logging.info("  %s", output_file)

    
    # output_files = FLAGS.output_file.split(",")
    # tf.logging.info("*** Writing to output files ***")
    # for output_file in output_files:
    #   tf.logging.info("  %s", output_file)

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    total_read = 0
    #rng = random.Random(FLAGS.random_seed)
    id = 0
    
    source_id_dir = save_dir + "/source_ids"
    if os.path.exists(source_id_dir):
        shutil.rmtree(source_id_dir) 
    os.mkdir(source_id_dir)
    
    source_mask_dir = save_dir + "/source_masks"
    if os.path.exists(source_mask_dir):
        shutil.rmtree(source_mask_dir)
    os.mkdir(source_mask_dir)
    
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

              if len(source_tokens) >= max_seq_length or len(target_tokens) >= max_seq_length:
                  tf.logging.info("ignore long sentence!")
                  continue

              instance = create_training_instance(source_tokens, target_tokens, max_seq_length, clip_to_max_len)
              if instance is None:
                  continue

              features = write_instance_to_file(writers[writer_index], instance, tokenizer, max_seq_length)

              writer_index = (writer_index + 1) % len(writers)
              total_written += 1

              for feature_name in features.keys():
                  feature = features[feature_name]
                  values = []
                  if feature.int64_list.value:
                      values = feature.int64_list.value
                  elif feature.float_list.value:
                      values = feature.float_list.value
                  tf.logging.info(
                      "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
                  list1 = np.array([str(x) for x in values]).astype(np.int32)
                  #print("+++++++++++++++++++++++++list shape is",list1.shape,list1.dtype)
                  if feature_name is "source_eos_ids" :
                      list1.tofile(source_id_dir + "/transform_input_data_"+ str(id) + ".bin")
                  elif feature_name is "source_eos_mask" :
                      list1.tofile(source_mask_dir + "/transform_input_data_"+ str(id) + ".bin")
              
              id = id + 1
    
    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)



if __name__ == "__main__":
  
    args = parser.parse_args()
    
    input_file = args.input_file
    vocab_file = args.vocab_file
    output_file = args.output_file
    max_seq_length = args.max_seq_length
    clip_to_max_len = args.clip_to_max_len
    num_splits = args.num_splits
    save_dir = args.save_dir
    
    create_infer_input_data(input_file, vocab_file, output_file, max_seq_length, clip_to_max_len, num_splits, save_dir)
