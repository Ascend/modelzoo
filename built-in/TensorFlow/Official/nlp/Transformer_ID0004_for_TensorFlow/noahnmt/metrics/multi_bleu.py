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

"""BLEU metric implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import re
import subprocess
import tempfile
import numpy as np

from six.moves import urllib
import tensorflow as tf

from noahnmt.utils import constant_utils


def calc_bleu(hypothesis_file, reference_file, lowercase=False, script=None):
  # if script provided, use script instead
  # take hyp_file and ref_file as input
  if script:
    # with open(hypothesis_file, "r") as read_pred:
    bleu_cmd = ["bash"] + [script] + [hypothesis_file] + [reference_file]
    try:
      bleu_out = subprocess.check_output(
          bleu_cmd, stderr=subprocess.STDOUT)
      bleu_score = float(bleu_out.decode("utf-8"))
    except Exception as error:
      tf.logging.warning("script returned non-zero exit code")
      tf.logging.warning(error)
      bleu_score = float(0.0)
    return bleu_score

  metrics_dir = os.path.dirname(os.path.realpath(__file__))
  bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", ".."))
  multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

  with open(hypothesis_file, "r") as read_pred:
    bleu_cmd = ["perl"] + [multi_bleu_path]
    if lowercase:
      bleu_cmd += ["-lc"]
    bleu_cmd += [reference_file]
    try:
      bleu_out = subprocess.check_output(
          bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
      bleu_out = bleu_out.decode("utf-8")
      bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
      bleu_score = float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        tf.logging.warning("multi-bleu.perl script returned non-zero exit code")
        tf.logging.warning(error.output)
      bleu_score = float(0.0)
  return bleu_score


def moses_multi_bleu(hypotheses, references, lowercase=False, script=None):
  """Calculate the bleu score for hypotheses and references
  using the MOSES ulti-bleu.perl script.

  Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script

  Returns:
    The BLEU score as a float32 value.
  """

  if np.size(hypotheses) == 0:
    return constant_utils.NP_FLOAT()(0.0)

  # Dump hypotheses and references to tempfiles
  delete = not sys.platform.startswith('win')
  hypothesis_file = tempfile.NamedTemporaryFile(delete=delete)
  hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
  hypothesis_file.write(b"\n")
  hypothesis_file.flush()
  reference_file = tempfile.NamedTemporaryFile(delete=delete)
  reference_file.write("\n".join(references).encode("utf-8"))
  reference_file.write(b"\n")
  reference_file.flush()

  # Calculate BLEU using multi-bleu script
  bleu_score = calc_bleu(
      hypothesis_file.name,
      reference_file.name,
      lowercase=lowercase,
      script=script)

  # Close temp files
  hypothesis_file.close()
  reference_file.close()

  if not delete:
    os.unlink(hypothesis_file.name)
    os.unlink(reference_file.name)

  return constant_utils.NP_FLOAT()(bleu_score)
