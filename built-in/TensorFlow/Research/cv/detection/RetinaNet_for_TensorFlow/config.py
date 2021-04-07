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

from train import data_path
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
         "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

A = 9  #number of anchor
K = 20  #number of class

area = [32, 64, 128, 256, 512]
aspect_ratio = [0.5, 2.0, 1.0]
scales = [1.0, 1.26, 1.59]

BATCH_SIZE = 2
IMG_H = 512
IMG_W = 512
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.001

#XML_PATH = "./VOCdevkit/VOC2007/Annotations/"
#IMG_PATH = "./VOCdevkit/VOC2007/JPEGImages/"
XML_PATH = data_path + "/VOCdevkit/VOC2007/Annotations/"
IMG_PATH = data_path + "/VOCdevkit/VOC2007/JPEGImages/"
