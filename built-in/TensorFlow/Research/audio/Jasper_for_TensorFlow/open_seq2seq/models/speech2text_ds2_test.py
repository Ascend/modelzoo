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

import tensorflow as tf

from open_seq2seq.test_utils.test_speech_configs.ds2_test_config import \
    base_params, train_params, eval_params, base_model
from .speech2text_test import Speech2TextModelTests


class DS2ModelTests(Speech2TextModelTests):

  def setUp(self):
    self.base_model = base_model
    self.base_params = base_params
    self.train_params = train_params
    self.eval_params = eval_params

  def tearDown(self):
    pass

  def test_regularizer(self):
    return self.regularizer_test()

  def test_convergence(self):
    return self.convergence_test(5.0, 30.0, 0.1)

  def test_convergence_with_iter_size(self):
    return self.convergence_with_iter_size_test()

  def test_infer(self):
    return self.infer_test()

  def test_mp_collection(self):
    return self.mp_collection_test(14, 7)

  def test_levenshtein(self):
    return self.levenshtein_test()

  def test_maybe_functions(self):
    return self.maybe_functions_test()


if __name__ == '__main__':
  tf.test.main()
