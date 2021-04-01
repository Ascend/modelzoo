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
'''MusicTaggerTrainer'''

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore import nn
import os
import mindspore.dataset as ds
from musictagging.utils import create_dataset

class BCELoss(nn.Cell):
    def __init__(self,record=None):
        super(BCELoss, self).__init__(record)
        self.sm_scalar = P.ScalarSummary()
        self.cast = P.Cast()
        self.record = record
        self.weight = None
        self.bce = P.BinaryCrossEntropy()

    def construct(self, input, target):
        target = self.cast(target, ms.float32)
        loss = self.bce(input, target, self.weight)
        if self.record:
            self.sm_scalar("loss", loss)
        return loss

def train(model, network, dataset_direct, filename, columns_list, 
          num_consumer = 4, batch=16, epoch=50, save_checkpoint_steps=2000, 
          keep_checkpoint_max=50, prefix="model", directory = './', ):

    config_ck = CheckpointConfig(save_checkpoint_steps = save_checkpoint_steps, 
                                keep_checkpoint_max = keep_checkpoint_max) 
    ckpoint_cb = ModelCheckpoint(prefix = prefix,
                                 directory = directory,
                                 config=config_ck) 
    data_train =  create_dataset(dataset_direct, filename, batch, epoch, columns_list, num_consumer)


    model.train(epoch,data_train,
                callbacks=[ckpoint_cb, LossMonitor(per_print_times=100)],  
                dataset_sink_mode=False)
