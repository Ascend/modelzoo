# Copyright (c) 2018 NVIDIA Corporation
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
import tensorflow as tf

from .loss import Loss

class WavenetLoss(Loss):

  def __init__(self, params, model, name="wavenet_loss"):
    super(WavenetLoss, self).__init__(params, model, name)
    self._n_feats = self._model.get_data_layer().params["num_audio_features"]

  def get_required_params(self):
    return {}

  def get_optional_params(self):
    return {}

  def _compute_loss(self, input_dict):
    """
    Computes the cross-entropy loss for WaveNet.

    Args:
      input_dict (dict):
        * "decoder_output": array containing: [
          * logits: predicted output signal as logits
          * outputs: array containing: [
            * ground truth signal as encoded labels
            * mu-law decoded audio
          ]
        ]
    """

    prediction = tf.cast(input_dict["decoder_output"]["logits"], tf.float32)
    target_output = input_dict["decoder_output"]["outputs"][0]

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, 
        labels=target_output
    )
    loss = tf.reduce_mean(loss)

    return loss
