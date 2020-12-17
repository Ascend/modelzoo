# Copyright (c) 2017 NVIDIA Corporation
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
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import io

from six.moves import range


def pad_vocab_to_eight(vocab):
  """Pads vocabulary so that it is divisible by 8.

  Args:
    vocab (dict): vocabulary in the form token->id

  Returns:
    dict: vocab with new tokens added if necessary, such that the total
    vocab size is divisible by 8.
  """
  v_len = len(vocab)
  if v_len % 8 == 0:
    return vocab
  for id_add in range(0, 8 - v_len % 8):
    vocab['<$'+str(id_add)+'$>'] = v_len + id_add
  return vocab


def load_pre_existing_vocabulary(path, min_idx=0, read_chars=False):
  """Loads pre-existing vocabulary into memory.

  The vocabulary file should contain a token on each line with optional
  token count on the same line that will be ignored. Example::

    a 1234
    b 4321
    c 32342
    d
    e
    word 234

  Args:
    path (str): path to vocabulary.
    min_idx (int, optional): minimum id to assign for a token.
    read_chars (bool, optional): whether to read only the
        first symbol of the line.

  Returns:
     dict: vocabulary dictionary mapping tokens (chars/words) to int ids.
  """
  idx = min_idx
  vocab_dict = {}
  with io.open(path, newline='', encoding='utf-8') as f:
    for line in f:
      # ignoring empty lines
      if not line or line == '\n':
        continue
      if read_chars:
        token = line[0]
      else:
        token = line.rstrip().split('\t')[0]
      vocab_dict[token] = idx
      idx += 1
  return vocab_dict
