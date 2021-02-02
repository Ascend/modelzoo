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
import os
import easydict

edict = easydict.EasyDict

cfg = edict()
cfg.FILTER_MIN_AREA = 1e-4
cfg.EPSILON_RATIO = 0.001

cfg.EVAL = edict()
cfg.EVAL.NUM_READERS = 1
cfg.EVAL.IMG_DIR = './datasets/total_text/test_images'
cfg.EVAL.LABEL_DIR = './datasets/total_text/test_gts'
