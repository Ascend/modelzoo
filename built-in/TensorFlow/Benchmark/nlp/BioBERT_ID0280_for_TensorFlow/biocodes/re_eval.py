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
import numpy as np
import pandas as pd
import sklearn.metrics
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--task', type=str,  default="binary", help='default:binary, possible other options:{chemprot}')
args = parser.parse_args()


testdf = pd.read_csv(args.answer_path, sep="\t", index_col=0)
preddf = pd.read_csv(args.output_path, sep="\t", header=None)


# binary
if args.task == "binary":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    pred_prob_one = [v[1] for v in pred]

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"])
    results = dict()
    results["f1 score"] = f[1]
    results["recall"] = r[1]
    results["precision"] = p[1]
    results["specificity"] = r[0]

# chemprot
# micro-average of 5 target classes
# see "Potent pairing: ensemble of long short-term memory networks and support vector machine for chemical-protein relation extraction (Mehryary, 2018)" for details
if args.task == "chemprot":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    str_to_int_mapper = dict()

    for i,v in enumerate(sorted(testdf["label"].unique())):
        str_to_int_mapper[v] = i
    test_answer = [str_to_int_mapper[v] for v in testdf["label"]]

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, labels=[0,1,2,3,4], average="micro")
    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p

for k,v in results.items():
    print("{:11s} : {:.2%}".format(k,v))


