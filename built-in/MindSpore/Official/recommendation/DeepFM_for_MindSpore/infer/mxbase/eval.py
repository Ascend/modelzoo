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
To generate the evaluation metric AUC.
Usage example: python eval.py --pred_file preds.txt --label_file label.txt
"""

from numpy import genfromtxt
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--pred_file',
    default='../data/preds_sdk.txt',
    help='preditions')
parser.add_argument(
    '--label_file',
    default='../data/label.txt',
    help='ground truth')

args = parser.parse_args()

print(f"loading pred_file: {args.pred_file}")
preds = genfromtxt(args.pred_file, delimiter=',')

print(f"loading label_file: {args.label_file}")
labels = genfromtxt(args.label_file, delimiter=',')

auc = roc_auc_score(labels, preds)

print('AUC: ', auc)
