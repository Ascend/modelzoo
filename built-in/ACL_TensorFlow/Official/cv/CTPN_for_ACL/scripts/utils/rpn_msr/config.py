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
class Config:
    EPS = 1e-14
    RPN_CLOBBER_POSITIVES = False
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 300
    RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    RPN_POSITIVE_WEIGHT = -1.0

    RPN_PRE_NMS_TOP_N = 12000
    RPN_POST_NMS_TOP_N = 1000
    RPN_NMS_THRESH = 0.7
    RPN_MIN_SIZE = 8
