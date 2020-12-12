#!/bin/bash

#
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

inferRet_folder="../model1_dev_0_chn_0_results/bert/"
real_file="../datasets/bert_tnews_origin/dev.json"
label_file="../datasets/bert_tnews_origin/labels.json"
for para in $*
do
        if [[ $para == --inferRet_folder* ]];then
                inferRet_folder=`echo ${para#*=}`
        elif [[ $para == --real_file* ]];then
                real_file=`echo ${para#*=}`
        elif [[ $para == --label_file* ]];then
                label_file=`echo ${para#*=}`
        fi
done

python3 calc_bert_accuracy.py $inferRet_folder $real_file $label_file

