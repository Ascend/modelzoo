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
import numpy as np
from scipy.io.wavfile import write

from .encoder_decoder import EncoderDecoderModel

def save_audio(signal, logdir, step, sampling_rate, mode):
  signal = np.float32(signal)
  file_name = '{}/sample_step{}_{}.wav'.format(logdir, step, mode)
  if logdir[0] != '/':
    file_name = "./" + file_name
  write(file_name, sampling_rate, signal)

class Text2SpeechWavenet(EncoderDecoderModel):

  @staticmethod
  def get_required_params():
    return dict(
        EncoderDecoderModel.get_required_params(), **{}
    )

  def __init__(self, params, mode="train", hvd=None):
    super(Text2SpeechWavenet, self).__init__(params, mode=mode, hvd=hvd)

  def maybe_print_logs(self, input_values, output_values, training_step):
    save_audio(
        output_values[1][-1],
        self.params["logdir"],
        training_step,
        sampling_rate=22050,
        mode="train"
    )
    return {}

  def evaluate(self, input_values, output_values):
    return output_values[1][-1]

  def finalize_evaluation(self, results_per_batch, training_step=None):
    save_audio(
        results_per_batch[0],
        self.params["logdir"],
        training_step,
        sampling_rate=22050,
        mode="eval"
    )
    return {}

  def infer(self, input_values, output_values):
    return output_values[1][-1]

  def finalize_inference(self, results_per_batch, output_file):
    return {}
