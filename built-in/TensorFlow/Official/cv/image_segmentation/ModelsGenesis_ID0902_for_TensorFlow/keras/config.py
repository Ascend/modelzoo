#
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
#
from npu_bridge.npu_init import *
import os
import shutil
import sys

class models_genesis_config:
    model = "Vnet"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    # data
    data = sys.argv[1]
    train_fold=[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = int(sys.argv[3])
    patience = 50
    lr = 1e0

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    sample_path = "pair_samples"
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    shutil.rmtree(os.path.join(sample_path, exp_name), ignore_errors=True)
    if not os.path.exists(os.path.join(sample_path, exp_name)):
        os.makedirs(os.path.join(sample_path, exp_name))
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

