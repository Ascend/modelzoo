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

"""
A collection of commonly used post-processing functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import unicodedata
import re


# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")


def _unescape_token(escaped_token):
  """Inverse of _escape_token().

  Args:
    escaped_token: a unicode string

  Returns:
    token: a unicode string
  """

  def match(m):
    if m.group(1) is None:
      return u"_" if m.group(0) == u"\\u" else u"\\"

    try:
      return six.unichr(int(m.group(1)))
    except (ValueError, OverflowError) as _:
      return u"\u3013"  # Unicode for undefined character.

  trimmed = escaped_token[:-1] if escaped_token.endswith("_") else escaped_token
  return _UNESCAPE_REGEX.sub(match, trimmed)


def strip_bpe(text):
  """Deodes text that was processed using BPE from
  https://github.com/rsennrich/subword-nmt"""
  return text.replace("@@ ", "").strip()


def decode_sentencepiece(text):
  """Decodes text that uses https://github.com/google/sentencepiece encoding.
  Assumes that pieces are separated by a space"""
  return "".join(text.split()).replace("▁", " ").strip()


def decode_bert(text):
  """Decodes text that uses https://github.com/google/sentencepiece encoding.
  Assumes that pieces are separated by a space"""
  # return "".join(text.split()).replace("##", " ").strip()
  return text.replace(" ##", "").strip()


def decode_t2t(text):
  """Decodes text that uses https://github.com/google/tensor2tensor encoding.
  Assumes that pieces are separated by a space"""
  text = "".join(text.split()).replace("_", " ").replace("▁", " ")
  return " ".join([_unescape_token(x) for x in text.split()])


def slice_text(text,
               eos_token="<s>",
               sos_token="</s>"):
  """Slices text from <s> to </s>, not including
  these special tokens.
  """
  eos_index = text.find(eos_token)
  text = text[:eos_index] if eos_index > -1 else text
  sos_index = text.find(sos_token)
  text = text[sos_index+len(sos_token):] if sos_index > -1 else text
  return text.strip()


def get_postproc_fn(name):
  if "bpe" == name:
    return strip_bpe
  elif "spm" == name:
    return decode_sentencepiece
  elif "t2t" == name:
    return decode_t2t
  elif "bert" == name:
    return decode_bert
  else:
    raise ValueError("Postproc fn not found: %s" % name)