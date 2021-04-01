# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''demo'''

import librosa as lr
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from musictagging.model import MusicTaggerCNN
import argparse
import numpy as np
import os
from mindspore import Tensor, context
from scripts.data_conversion import compute_melgram

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict raw audio file')
    parser.add_argument('--npu', type=int, help='device ID', default = 0)
    parser.add_argument('--model_dir', type=str, help='model directory')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--audio_file', type=str, help='audio file')
    
    args = parser.parse_args()

    context.set_context(device_target='Ascend',mode = context.GRAPH_MODE, device_id = args.npu)
    network = MusicTaggerCNN()
    param_dict = load_checkpoint(args.model_dir + '/' + args.model_name)
    load_param_into_net(network, param_dict)
    audio_feature = compute_melgram(args.audio_file, save_npy = False)
    if audio_feature.shape[3] < 1366:
        print("The audio clip is too short for tagging. The minimum length fur tagging is 29.1 s")
    else:
        audio_feature = audio_feature[:,:,:,-1366:]
        pred_tag = network(Tensor(audio_feature)).asnumpy()
        prediction = []
        with open("tag.txt","rb") as f:
            for i in range(50):
                prediction.append([pred_tag[0][i], f.readline().strip().decode('utf-8')])
        prediction = sorted(prediction)[::-1]
        print("=" * 10, "Probability", "=" * 10)
        for i in range(50):
            if prediction[i][0] > 0.000005:
                print("{0:20}{1:.5f}".format(prediction[i][1],prediction[i][0]))
