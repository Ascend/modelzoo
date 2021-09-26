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

import os
from sklearn.metrics import  average_precision_score, roc_auc_score
import numpy as np
import sys

def aucPerformance(mse, labels):
    """
    :param mse:
    :param labels:
    :return:
    """
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def eval_om(label_dir, om_output_dir):
    """
    :param label_dir:
    :param om_output_dir:
    :return:
    """
    label, score = read_directory(label_dir, om_output_dir)
    aucPerformance(score, label)


def read_directory(label_dir, om_output_dir):
    """
    :param label_dir:
    :param om_output_dir:
    :return:
    """
    # get label bin files
    labels = os.listdir(label_dir)
    labels.sort()
    labels_data = list()

    # get om output files
    outputs = os.listdir(om_output_dir)
    outputs.sort()
    outputs_data = list()

    for i in range(len(outputs)):
        label_data = np.fromfile(os.path.join(label_dir, labels[i]), dtype=np.int32)
        labels_data.extend(label_data)
        output_data = np.fromfile(os.path.join(om_output_dir, outputs[i]), dtype=np.float32)
        outputs_data.extend(output_data)
    return labels_data, outputs_data

gt_dir = sys.argv[1]
predict_dir = sys.argv[2]
eval_om(gt_dir, predict_dir)
