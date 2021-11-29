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

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import codecs
import tempfile
import collections
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

# from ..utils import misc_utils as utils


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class SpecialVocab(
    collections.namedtuple("SpecialVocab",
                           ["unk", "sos", "eos"])):
  pass

class VocabInfo(
    collections.namedtuple("VocbabInfo",
                           ["path", "vocab_size", "special_vocab"])):
  """Convenience structure for vocabulary information.
  """
  pass

def get_special_vocab(vocabulary_size, special_vocabs):
  """Returns the `SpecialVocab` instance for a given vocabulary size.
  """
  return SpecialVocab(
      unk=special_vocabs["unk"], 
      sos=special_vocabs["sos"],
      eos=special_vocabs["eos"])

def get_vocab_info(vocab_path, special_vocabs, pad_to_eight=False):
  """Creates a `VocabInfo` instance that contains the vocabulary size and
    the special vocabulary for the given file.

  Args:
    vocab_path: Path to a vocabulary file with one word per line.

  Returns:
    A VocabInfo tuple.
  """

  """vocab_size, vocab_file = check_vocab(
      vocab_file=vocab_path, 
      special_vocabs=special_vocabs,
      pad_to_eight=pad_to_eight)"""
  vocab_size = count_vocab(vocab_path)
  vocab_file = vocab_path
  spec_vocab = get_special_vocab(vocab_size, special_vocabs)
  spec_vocab = SpecialVocab(
      unk=0, 
      sos=1,
      eos=2)
  return VocabInfo(path=vocab_file,
                   vocab_size=vocab_size,
                   special_vocab=spec_vocab)


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def count_vocab(vocab_file):
  if tf.gfile.Exists(vocab_file):
    vocab2id = collections.OrderedDict()
    vocab_size = 0
    # with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    #   vocab_size = 0
    #   for word in f:
    #     items = word.strip().split() + [" "]
    #     v = items[0]
    #     if v == " ":
    #       tf.logging.warning("empty vocab entry")
    #     vocab2id[v] = vocab_size
    #     vocab_size += 1
    with tf.gfile.GFile(vocab_file, "r") as reader:
      while True:
        token = convert_to_unicode(reader.readline())
        if not token:
          break
        vocab_size += 1
    if vocab_size % 16 > 0:
      vocab_size = (16-vocab_size%16) + vocab_size
    return vocab_size
  else:
    raise ValueError("vocab_file does not exist.")


def check_vocab(vocab_file, special_vocabs, pad_to_eight):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  global UNK, SOS, EOS, UNK_ID

  if tf.gfile.Exists(vocab_file):
    tf.logging.info("# Vocab file %s exists\n" % vocab_file)
    vocab = []
    vocab_size = 0
    # with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    #   vocab_size = 0
    #   for word in f:
    #     vocab_size += 1
    #     items = word.strip().split() + [" "]
    #     if items[0] == " ":
    #       tf.logging.warning("empty vocab entry")
    #     vocab.append(items[0])
    with tf.gfile.GFile(vocab_file, "r") as reader:
      while True:
        token = convert_to_unicode(reader.readline())
        if not token:
          break
        token=token.strip()
        if len(token) == 0:
          continue
        token = token.split()[0]
        vocab.append(token)
        vocab_size += 1

    # Verify if the vocab starts with unk, sos, eos
    # If not, prepend those tokens & generate a new vocab file
    if special_vocabs:
      unk = special_vocabs.get("unk", UNK)
      sos = special_vocabs.get("sos", SOS)
      eos = special_vocabs.get("eos", EOS)

      flag = False
      for tok in [eos, sos, unk]:
        if tok not in vocab:
          tf.logging.info("Special token %s not in vocab. Added!" % tok)
          vocab = [tok] + vocab
          flag = True
      
      for key, tok in [("eos",eos), ("sos",sos), ("unk",unk)]:
        special_vocabs[key] = vocab.index(tok)
      
      UNK_ID = special_vocabs["unk"]
      UNK = unk
      EOS = eos
      SOS = sos

      
      if flag:
        # new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
        with tempfile.NamedTemporaryFile() as tmpfile:
          new_vocab_file = tmpfile.name
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
        vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file does not exist: %s" % vocab_file)

  vocab_size = len(vocab)
  # pad_to_eight to speedup float16 calculation
  if pad_to_eight and vocab_size % 8 > 0:
    vocab_size = vocab_size + 8 - vocab_size % 8
  return vocab_size, vocab_file

def read_vocab(vocab_file):
  if tf.gfile.Exists(vocab_file):
    vocab2id = collections.OrderedDict()
    vocab_size = 0
    # with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    #   vocab_size = 0
    #   for word in f:
    #     items = word.strip().split() + [" "]
    #     v = items[0]
    #     if v == " ":
    #       tf.logging.warning("empty vocab entry")
    #     vocab2id[v] = vocab_size
    #     vocab_size += 1
    with tf.gfile.GFile(vocab_file, "r") as reader:
      while True:
        token = convert_to_unicode(reader.readline())
        if not token:
          break
        token=token.strip()
        if len(token) == 0:
          continue
        token = token.split()[0]
        vocab2id[token] = vocab_size
        vocab_size += 1
    return vocab2id
  else:
    raise ValueError("vocab_file does not exist.")


def _build_hash_table(vocab2id, default_value, reverse=False):
  keys = list(vocab2id.keys())
  values = list(vocab2id.values())

  key_dtype = None
  if reverse:
    keys, values = values, keys
    key_dtype = tf.int64
  
  init = lookup_ops.KeyValueTensorInitializer(
      keys=keys, values=values, key_dtype=key_dtype)
  hash_table = lookup_ops.HashTable(init, default_value)
  return hash_table


def create_vocab_tables(vocab_file):
  """Creates vocab tables."""
  # this implementation is c++-friendly in freezed model
  vocab2id = read_vocab(vocab_file)
  
  vocab_table = _build_hash_table(
      vocab2id, UNK_ID)
  reverse_vocab_table = _build_hash_table(
      vocab2id, UNK, reverse=True)
  
  return vocab_table, reverse_vocab_table
