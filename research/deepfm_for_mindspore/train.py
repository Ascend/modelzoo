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
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from src.deepfm import DeepFMModel, ModelBuilder
from src.config import DataConfig, ModelConfig, TrainConfig
from src.metric import AUCMetric
from src.datasets import create_dataset
from src.callback import EvalCallBack, LossCallBack


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--do_eval', type=bool, default=True, help='Do evaluation or not.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt, _ = parser.parse_known_args()

device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(enable_task_sink=True, device_id=device_id)


if __name__ == '__main__':

    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    ds_train = create_dataset(args_opt.dataset_path, train_mode=True, 
                              epochs=train_config.train_epochs,
                              batch_size=train_config.batch_size)
    
    model_builder = ModelBuilder(ModelConfig, TrainConfig)
    train_net, eval_net = model_builder.get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    callback_list = []
    loss_callback = LossCallBack(loss_file_path=os.path.join(train_config.output_path, train_config.loss_file_name))
    callback_list.append(loss_callback)
    if train_config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=train_config.save_checkpoint_steps,
                                    keep_checkpoint_max=train_config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=train_config.ckpt_file_name_prefix, 
                                    directory=train_config.ckpt_path, 
                                    config=config_ck)
        callback_list.append(ckpt_cb)

    if args_opt.do_eval:
        ds_eval = create_dataset(args_opt.dataset_path, train_mode=False, 
                            epochs=train_config.train_epochs,
                            batch_size=train_config.batch_size)
        eval_callback = EvalCallBack(model, ds_eval, auc_metric,
                                    eval_file_path=os.path.join(train_config.output_path, train_config.eval_file_name))
        callback_list.append(eval_callback)

    
    model.train(train_config.batch_size, ds_train, callbacks=callback_list)


