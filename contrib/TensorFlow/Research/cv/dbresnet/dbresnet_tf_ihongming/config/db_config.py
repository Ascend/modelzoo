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
import os
import easydict

edict = easydict.EasyDict

cfg = edict()
cfg.root = './logs'
cfg.PLATFORM = 'NPU'

cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True

# 训练
cfg.TRAIN.VIS_GPU = '0'
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.MAX_STEPS = 187200
cfg.TRAIN.LEARNING_RATE = 0.007
cfg.TRAIN.LOSS_ALPHA = 1.0
cfg.TRAIN.LOSS_BETA = 10.0

# 存储目录
cfg.TRAIN.SAVE_CHECKPOINT_STEPS = 3000
cfg.TRAIN.SAVE_SUMMARY_STEPS = 200
cfg.TRAIN.SAVE_MAX = 1000
cfg.TRAIN.DATA_DIR = './datasets/total_text'
cfg.TRAIN.TRAIN_LOGS = os.path.join(cfg.root, 'tf_logs')
cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR = os.path.join(cfg.root, 'ckpt')

# dataload
cfg.MEANS = [122.67891434, 116.66876762, 104.00698793]
cfg.EPSILON_RATIO = 0.001
cfg.SHRINK_RATIO = 0.4
cfg.THRESH_MIN = 0.3
cfg.THRESH_MAX = 0.7
cfg.TRAIN.IMG_SIZE = 640
cfg.TRAIN.MIN_TEXT_SIZE = 8

cfg.LR = 'paper_decay'
cfg.ADAM_DECAY_STEP = 10000
cfg.ADAM_DECAY_RATE = 0.9

cfg.TRAIN.OPT = 'sgd'
cfg.TRAIN.MOVING_AVERAGE_DECAY = 0.9
cfg.lr= 'exponential_decay'

cfg.TRAIN.RESTORE = None
cfg.TRAIN.RESTORE_CKPT_PATH = None
cfg.TRAIN.model_checkpoint_path= None
cfg.TRAIN.PRETRAINED_MODEL_PATH = None

cfg.EVAL = edict()
cfg.EVAL.NUM_READERS = 1
cfg.EVAL.IMG_DIR = '../datasets/total_text/test_images'
cfg.EVAL.LABEL_DIR = '../datasets/total_text/test_gts'

cfg.K = 50
cfg.BACKBONE = 'resnet_v1_50'
cfg.ASPP_LAYER = False
cfg.MEANS = [122.67891434, 116.66876762, 104.00698793]
cfg.FILTER_MIN_AREA = 1e-4
