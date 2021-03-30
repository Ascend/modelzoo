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
'''test'''

from mindspore import context
from musictagging.model import MusicTaggerCNN
from musictagging.MusicTaggerValidator import validation
import argparse
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model_dir', type=str, help='model directory')
    parser.add_argument('--npu', type=int, help='device ID', default = 0)
    parser.add_argument('--batch', type=int, help='batch size', default = 32)
    parser.add_argument('--data_dir', type=str, help='path to train data')
    parser.add_argument('--filename', type=str, help='name of train data file')
    parser.add_argument('--num_consumer', type=int, help='number of consumer', default = 4)
    parser.add_argument('--model_name', type=str, help='name of model')
    args = parser.parse_args()

    context.set_context(device_target='Ascend',
                        mode = context.GRAPH_MODE, 
                        device_id = args.npu)
    network = MusicTaggerCNN()
    network.set_train(False)
    auc = validation(network, 
                     args.model_dir + "/" + args.model_name, 
                     args.data_dir, 
                     args.filename, 
                     args.num_consumer, 
                     args.batch)

    print("=" * 10 + "Validation Peformance" + "=" * 10)
    print("AUC: {:.5f}".format(auc))
