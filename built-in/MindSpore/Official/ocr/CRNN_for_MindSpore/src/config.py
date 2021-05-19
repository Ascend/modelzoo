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
"""Network parameters."""
from easydict import EasyDict

label_dict = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
# use for low case number
config_default = {
    "max_text_length": 23,
    "image_width": 100,
    "image_height": 32,
    "batch_size": 4,
    "epoch_size": 10,
    "hidden_size": 256,
    "learning_rate": 0.02,
    "momentum": 0.95,
    "nesterov": True,
    "save_checkpoint": True,
    "save_checkpoint_steps": 1000,
    "keep_checkpoint_max": 30,
    "save_checkpoint_path": "./",
    "class_num": len(label_dict)+1,
    "input_size": 512,
    "use_dropout": True,
    "blank": len(label_dict)
}

config_add = {"num_step": config_default["max_text_length"] + 1}

config1 = EasyDict({**config_default, **config_add})
