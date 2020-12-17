# Copyright (c) 2018 NVIDIA Corporation
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
# Copyright 2020 Huawei Technologies Co., Ltd
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
'''
This file implements function to calcuate basic metrics.
'''
import numpy as np
import tensorflow as tf

def true_positives(labels, preds):
  return np.sum(np.logical_and(labels, preds)) 

def accuracy(labels, preds):
  return np.sum(np.equal(labels, preds)) / len(preds)

def recall(labels, preds):
  return true_positives(labels, preds) / np.sum(labels)

def precision(labels, preds):
  return true_positives(labels, preds) / np.sum(preds)

def f1(labels, preds):
  rec = recall(labels, preds)
  pre = precision(labels, preds)
  if rec == 0 or pre == 0:
    return 0
  return 2 * rec * pre / (rec + pre)
