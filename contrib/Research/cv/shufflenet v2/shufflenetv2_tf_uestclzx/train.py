# coding=utf-8

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

import tensorflow as tf
import os
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
from npu_bridge.estimator.npu.npu_config import NPURunConfig 
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
#from mxnet.model import save_checkpoint
import moxing as mox
import math
mox.file.shift('os', 'mox')

tf.flags.DEFINE_string('data_url', " ", 'dataset path')
tf.flags.DEFINE_string('train_url', " ", 'train output path')
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity('INFO')

"""
The purpose of this script is to train a network.

To use it just run:
python train.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
BATCH_SIZE = 128  # batch_size for train data
VALIDATION_BATCH_SIZE = 128  # batch_size for eval data
NUM_EPOCHS = 160  # set 166 for 1.0x version

Epochs_between_evals = 5
TRAIN_DATASET_SIZE = 1281144  # the length of the imgnet12_train dataset after preprocessing
EVAL_DATASET_SIZE = 49999  # the length of the imgnet12_val dataset after preprocessing
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
"""
配置数据集路径
"""
# 训练集
Cache_Path = '/cache/data'
OBS_TRAIN_DATA_Path = os.path.join(FLAGS.data_url, 'train')
TMP_TRAIN_DATA_Path = os.path.join(Cache_Path,'train')
mox.file.copy_parallel(src_url=OBS_TRAIN_DATA_Path,dst_url=TMP_TRAIN_DATA_Path)
# 测试集
OBS_VAL_DATA_Path = os.path.join(FLAGS.data_url,'val')
TMP_VAL_DATA_Path = os.path.join(Cache_Path, 'val')
mox.file.copy_parallel(src_url=OBS_VAL_DATA_Path, dst_url=TMP_VAL_DATA_Path)
# 配置保存模型文件的路径
OBS_MODEL_Path = FLAGS.train_url
TMP_MODEL_Path = '/cache/model'
# 配置验证log保存的路径
OBS_EVALLOG_PATH = os.path.join(FLAGS.data_url, 'txt')
TMP_EVALLOG_PATH = '/cache/log'
mox.file.copy_parallel(src_url=OBS_EVALLOG_PATH, dst_url=TMP_EVALLOG_PATH)
# 配置预训练模型的路径
PRETRAINED_MODE_PATH = os.path.join(FLAGS.data_url, 'pretrained-model')
mox.file.copy_parallel(PRETRAINED_MODE_PATH, TMP_MODEL_Path)

PARAMS = {
    'train_dataset_path': TMP_TRAIN_DATA_Path,  # '/cache/data/',
    'val_dataset_path': TMP_VAL_DATA_Path,  # '/cache/data/',
    'weight_decay': 4e-5,
    'initial_learning_rate': 0.06,  # 0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-5,
    'model_dir': TMP_MODEL_Path,
    'num_classes': 1000,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}

def get_input_fn(is_training, num_epochs):

    dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    print("-----------------------------------",filenames)
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]
    batch_size = BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE

    def input_fn():
        pipeline = Pipeline(
            filenames, is_training,
            batch_size=batch_size,
            num_epochs=num_epochs
        )

        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE

run_config = NPURunConfig(
    model_dir=PARAMS['model_dir'], 
    session_config=session_config,
    keep_checkpoint_max=10,
    save_summary_steps=500,
    save_checkpoints_steps=50044,
    precision_mode='allow_mix_precision',
)

estimator =NPUEstimator(model_fn,
                        model_dir=PARAMS['model_dir'],
                        config=run_config,
                        params=PARAMS
                        )

n_loops = math.ceil(NUM_EPOCHS / Epochs_between_evals)
schedule = [Epochs_between_evals for _ in range(int(n_loops))]
schedule[-1] = NUM_EPOCHS - sum(schedule[:-1])  # over counting.

for cycle_index, num_train_epochs in enumerate(schedule):
    tf.compat.v1.logging.info('***********************************Starting cycle: %d/%d', cycle_index,int(n_loops))
    if num_train_epochs:
        # Since we are calling estimaor.train immediately in each loop
        estimator.train(input_fn=get_input_fn(True, num_train_epochs), hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])
    mox.file.copy_parallel(src_url=TMP_MODEL_Path, dst_url=OBS_MODEL_Path)
mox.file.copy_parallel(src_url=TMP_EVALLOG_PATH, dst_url=OBS_MODEL_Path)
print("Done!!!!!!!!!")