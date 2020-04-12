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
"""train_criteo."""
import os
import argparse
import random
import sys
import numpy as np

from mindspore import context
from mindspore.train.model import Model
from src.deepfm import DeepFMModel, ModelBuilder
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.config import DataConfig, ModelConfig, TrainConfig
from src.metric import AUCMetric
from src.datasets import create_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt, _ = parser.parse_known_args()

device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(enable_task_sink=True, device_id=device_id)


if __name__ == '__main__':

    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    ds_eval = create_dataset(args_opt.dataset_path, train_mode=False, 
                            epochs=1, batch_size=train_config.batch_size)
    
    model_builder = ModelBuilder(ModelConfig, TrainConfig)
    train_net, eval_net = model_builder.get_train_eval_net()

    train_net.set_train()
    eval_net.set_train(False)

    auc_metric = AUCMetric()

    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(eval_net, param_dict)

    res = model.eval(ds_eval)
    print("=="*5 + "result:", res)

    