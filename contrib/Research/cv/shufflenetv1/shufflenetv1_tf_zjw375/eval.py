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


tf.flags.DEFINE_string('data_url', " ", 'dataset path')
tf.flags.DEFINE_string('train_url', " ", 'train output path')
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity('INFO')

"""
The purpose of this script is to eval a checkpoint model.

To use it just run:
python eval.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
BATCH_SIZE = 96
VALIDATION_BATCH_SIZE = 96
NUM_EPOCHS = 200  # set 166 for 1.0x version
EPOCHS_TO_RUN = 2  # the true number of epoch to run

Epochs_between_evals = 2
TRAIN_DATASET_SIZE = 1281144 #共1281167
EVAL_DATASET_SIZE = 49999 #共50000
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)

# 配置路径
TMP_TRAIN_DATA_Path = './dataset/train' # 训练集
TMP_VAL_DATA_Path = './dataset/val' # 测试集
PRETRAINED_MODE_PATH = './model' # 预训练模型
TMP_MODEL_Path = './model' # 训练生成模型

PARAMS = {
	'train_dataset_path': TMP_TRAIN_DATA_Path,
	'val_dataset_path': TMP_VAL_DATA_Path,
	'weight_decay': 4e-5,
	'initial_learning_rate': 0.03,
	'decay_steps': NUM_STEPS,
	'end_learning_rate': 1e-5,
	'model_dir': TMP_MODEL_Path,
	'num_classes': 1000,
	'group': 3,
	'dropout': 0.5,
	'complexity_scale_factor': 0.5  # set '1.0' for 1.0x version
}

def get_input_fn(is_training, num_epochs):

	dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
	filenames = os.listdir(dataset_path)
	filenames = [n for n in filenames if n.endswith('.tfrecords')]#筛除后缀不是.tfrecords的文件
	print("-----------------------------------", filenames)
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

#设置运行配置
session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE

run_config = NPURunConfig(
	model_dir=PARAMS['model_dir'],
	session_config=session_config,
	save_summary_steps=500,
	save_checkpoints_steps=1200,
	precision_mode='allow_mix_precision',
)

estimator =NPUEstimator(model_fn,
						model_dir=PARAMS['model_dir'],
						config=run_config,
						params=PARAMS
						)
#配置验证输出文件路径
log_path = '/val_result.txt'
f = open(log_path, 'w')

for cycle_index in range(1):

	tf.compat.v1.logging.info('====================================Starting to evaluate.')
	eval_results = estimator.evaluate(input_fn=get_input_fn(False, 1), hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])
	for key in sorted(eval_results.keys()):
	    tf.logging.info("%s = %s", key, str(eval_results[key]))
	    line = (key, '\t', str(eval_results[key]), '\n')
	    f.writelines(line)
	enter = '\n'
	f.writelines(enter)
f.close()
print("Done!!!!!!!!!")