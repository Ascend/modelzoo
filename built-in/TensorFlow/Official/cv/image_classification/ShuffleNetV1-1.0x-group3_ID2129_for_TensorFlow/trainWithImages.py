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
import tensorflow as tf
import os
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')
from PIL import Image
import numpy as np
"""
The purpose of this script is to train a network.
Evaluation will happen periodically.

To use it just run:
python train.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
NUM_EPOCHS = 133  # set 133 for 1.0x version
TRAIN_DATASET_SIZE = 20547
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
PARAMS = {
    'train_dataset_path': 'data/tzb1224_train/',
    'val_dataset_path': 'data/tzb1224_val/',
    'weight_decay': 4e-5,
    'initial_learning_rate': 0.0625,  #0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-6,
    'model_dir': 'models/tzb1219',
    'num_classes': 2,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}

session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
session_config.gpu_options.allow_growth = True
run_config = tf.estimator.RunConfig(save_summary_steps=0)
run_config = run_config.replace(
    model_dir=PARAMS['model_dir'],
    session_config=session_config,
    save_summary_steps=500,
    save_checkpoints_secs=300,
    log_step_count_steps=50)

imgl = os.listdir('train_images')
img = Image.open('train_images/' + imgl[0])
region = (145, 153, 481, 489)
img_ = img.crop(region)
img = img_.resize((224, 224), Image.NEAREST)
images = (np.array(img, dtype=np.float32) / 255.0).reshape(1, 224, 224,1)
for imgn in imgl:
    if imgn != imgl[0]:
        img = Image.open('train_images/' + imgn)
        img_ = img.crop(region)
        img = img_.resize((224, 224), Image.NEAREST)
        image = (np.array(img, dtype=np.float32) / 255.0).reshape(1, 224, 224,1)
        images = np.append(images, image, axis=0)
labels = np.ones((len(imgl)), dtype=np.int32)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(100).repeat().batch(batch_size, drop_remainder=True)

    # Return the dataset.
    return dataset

labels = {'labels':np.array(labels)}
features = {'images': np.array(images)}
fn = lambda: train_input_fn(features, labels, BATCH_SIZE)
estimator = tf.estimator.Estimator(model_fn, params=PARAMS, config=npu_run_config_init(run_config=run_config))

estimator.train(input_fn=fn, steps=10)

