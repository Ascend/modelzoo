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
'''MusicTaggerValidator'''

import pandas as pd
import numpy as np
from mindspore.dataset import GeneratorDataset
from sklearn import metrics
from mindspore import Tensor
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from musictagging.utils import create_dataset

def val(network, data_dir, filename, num_consumer = 4, batch = 32):
    data_train = create_dataset(data_dir, filename, 
                                32, 1, ['feature','label'], 
                                num_consumer)
    data_train = data_train.create_tuple_iterator()
    res_pred = []
    res_true = []
    for data, label in data_train:
        x = network(Tensor(data, ms.float32))
        res_pred.append(x.asnumpy())
        res_true.append(label)
    res_pred = np.concatenate(res_pred, axis = 0)
    res_true = np.concatenate(res_true, axis = 0)
    auc = metrics.roc_auc_score(res_true, res_pred)
    return auc

def validation(network, model_path, data_dir, filename, num_consumer, batch):
    param_dict = load_checkpoint(model_path)
    load_param_into_net(network, param_dict)
        
    auc = val(network, data_dir, filename, num_consumer, batch)
    return auc
