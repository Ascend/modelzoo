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

"""custom estimator for convinence"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf
from tensorflow.python.estimator.estimator import _check_hooks_type

# from tensorflow.contrib.offline_train.python.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator


class NMTEstimator(NPUEstimator):
  def __init__(self, *args, **kwargs):
    # set assert_fun before init to prevent error: 
    # Subclasses of Estimator cannot override members of Estimator
    tf.estimator.Estimator._assert_members_are_not_overridden = lambda self: None
    super(NMTEstimator, self).__init__(*args, **kwargs)
  
  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              mode=tf.estimator.ModeKeys.PREDICT,
              checkpoint_path=None,
              yield_single_examples=True,
              init_from_scaffold=False):
    """Yields predictions for given features.

    Changes:
      init_from_scaffold: added to allow custom checkpoint_loading
    """
    hooks = _check_hooks_type(hooks)
    if not init_from_scaffold:
      # Check that model has been trained.
      if not checkpoint_path:
        checkpoint_path = tf.train.latest_checkpoint(self._model_dir)
      if not checkpoint_path:
        raise ValueError('Could not find trained model in model_dir: {}.'.format(
            self._model_dir))
    else:
      checkpoint_path = None

    with tf.Graph().as_default() as g:
      tf.set_random_seed(self._config.tf_random_seed)
      self._create_and_assert_global_step(g)
      features, input_hooks = self._get_features_from_input_fn(
          input_fn, mode)
      estimator_spec = self._call_model_fn(
          features, None, mode, self.config)
      predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
      all_hooks = list(input_hooks)
      all_hooks.extend(hooks)
      all_hooks.extend(list(estimator_spec.prediction_hooks or []))
      with tf.train.MonitoredSession(
          session_creator=tf.train.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path,
              master=self._config.master,
              scaffold=estimator_spec.scaffold,
              config=self._session_config),
          hooks=all_hooks) as mon_sess:
        while not mon_sess.should_stop():
          preds_evaluated = mon_sess.run(predictions)
          if not yield_single_examples:
            yield preds_evaluated
          elif not isinstance(predictions, dict):
            for pred in preds_evaluated:
              yield pred
          else:
            for i in range(self._extract_batch_length(preds_evaluated)):
              yield {
                  key: value[i]
                  for key, value in six.iteritems(preds_evaluated)
              }