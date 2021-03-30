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
'''train'''

from mindspore import context, nn
from musictagging.model import MusicTaggerCNN
from musictagging.MusicTaggerTrainer import train, BCELoss
from mindspore.train import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import argparse
from mindspore.train.serialization import load_checkpoint, load_param_into_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_dir', type=str, help='model directory', default = "./")
    parser.add_argument('--npu', type=int, help='device ID', default = 0)
    parser.add_argument('--epoch', type=int, help='epoch number', default = 5)
    parser.add_argument('--batch', type=int, help='batch size', default = 32)
    parser.add_argument('--lr', type=float, help='learning rate', default = 0.0005)
    parser.add_argument('--ls', type=float, help='loss scale', default = 1024.0)
    parser.add_argument('--save_step', type=int, help='save check point every N step', default = 2000)
    parser.add_argument('--keep_max', type=int, help='keep maximum checkpoint number', default = 2000)
    parser.add_argument('--data_dir', type=str, help='path to train data')
    parser.add_argument('--filename', type=str, help='name of train data file')
    parser.add_argument('--num_consumer', type=int, help='number of consumer', default = 4)
    parser.add_argument('--prefix', type=str, help='prefix of model', default = "Music_Tagger")
    parser.add_argument('--model_name', type=str, help='preload model', default = "")
    
    args = parser.parse_args()

    context.set_context(device_target='Ascend',mode = context.GRAPH_MODE, device_id = args.npu)
    network = MusicTaggerCNN()
    if args.model_name != "":
        param_dict = load_checkpoint(args.model_dir + '/' + args.model_name)
        load_param_into_net(network, param_dict)
       
    net_loss = BCELoss()
    
    network.set_train(True)
    net_opt = nn.Adam(params=network.trainable_params(), learning_rate = args.lr,loss_scale = args.ls)

    loss_scale_manager=FixedLossScaleManager(loss_scale = args.ls, drop_overflow_update = False)
    model = Model(network, net_loss, net_opt, loss_scale_manager = loss_scale_manager)

    train(model = model, 
          network = network, 
          dataset_direct = args.data_dir,
          filename = args.filename,
          columns_list = ['feature','label'],
          num_consumer = args.num_consumer,
          batch = args.batch, 
          epoch = args.epoch,
          save_checkpoint_steps = args.save_step,
          keep_checkpoint_max = args.keep_max,
          prefix = args.prefix,
          directory = args.model_dir)
