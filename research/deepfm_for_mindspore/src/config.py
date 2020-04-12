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
"""
network config setting, will be used in train.py and eval.py
"""

class DataConfig:
    data_vocab_size = 184965
    train_num_of_parts = 21
    data_path = "/opt/npu/deepFM/test_criteo_data_0/"
    test_num_of_parts = 3
    batch_size = 1000
    data_field_size = 39
    version = "svm"

class ModelConfig:
    batch_size = DataConfig.batch_size
    data_field_size = DataConfig.data_field_size
    data_vocab_size = DataConfig.data_vocab_size
    data_emb_dim = 80
    deep_layer_args = [[400, 400, 512], "relu"]
    init_args = [-0.01, 0.01]
    weight_bias_init = ['normal', 'normal']
    keep_prob = 0.9


class TrainConfig:
    batch_size = DataConfig.batch_size
    l2_coef = 1e-6
    learning_rate = 1e-5
    epsilon = 1e-8
    loss_scale = 1024.0

    train_epochs = 15

    output_path = "./output/"

    save_checkpoint = True
    ckpt_file_name_prefix = "deepfm"
    ckpt_path = "checkpint"
    checkpoint_file_name = "deepfm-10_41258.ckpt"
    save_checkpoint_steps = 1
    keep_checkpoint_max = 15
    
    result_file_name = "run_results.log"
    eval_callback = True
    eval_file_name = result_file_name
    loss_callback = True
    loss_file_name = result_file_name

# 


