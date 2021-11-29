# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# This file contains the parameter used in train.py

from __future__ import division, print_function

from utils.misc_utils import parse_anchors, read_class_names
import math
import os
import modelarts.frozen_graph as fg

save_dir =          '/cache/training/'  # The directory of the weights to save.
log_dir =           '/cache/training/logs/'  # The directory to store the tensorboard log files.
progress_log_path = '/cache/training/train.log'  # The path to record the training progress.

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

work_path = '/cache/user-job-dir/YoloV3_for_TensorFlow'

### Some paths
train_file =        os.path.join(work_path, './modelarts/coco2014_trainval_modelarts.txt')  # The path of the training txt file.
val_file =          os.path.join(work_path, './modelarts/coco2014_minival_modelarts.txt')  # The path of the validation txt file.
restore_path =      os.path.join(work_path, './data/darknet_weights/darknet53.ckpt')  # The path of the weights to restore.
anchor_path =       os.path.join(work_path, './data/yolo_anchors.txt')  # The path of the anchor txt file.
class_name_path =   os.path.join(work_path, './data/coco.names')  # The path of the class names.

fg.transform_images_path(os.path.join(work_path, './data/coco2014_trainval.txt'), train_file)
fg.transform_images_path(os.path.join(work_path, './data/coco2014_minival.txt'), val_file)
### Distribution setting
num_gpus = int(os.environ['RANK_SIZE'])
iterations_per_loop = 10

### Training releated numbersls

batch_size = 32
img_size = [416, 416]  # Images will be resized to `img_size` and fed to the network, size format: [width, height]
letterbox_resize = True  # Whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
total_epoches = 1
train_evaluation_step = 1000  # Evaluate on the training batch after some steps.
val_evaluation_epoch = 2  # Evaluate on the whole validation dataset after some epochs. Set to None to evaluate every epoch.
save_epoch = 10  # Save the model after some epochs.
batch_norm_decay = 0.99  # decay in bn ops
weight_decay = 5e-4  # l2 weight decay
global_step = 0  # used when resuming training

### tf.data parameters
num_threads = 8  # Number of threads for image processing used in tf.data pipeline.
prefetech_buffer = batch_size * 4   # Prefetech_buffer used in tf.data pipeline.

### Learning rate and optimizer
optimizer_name = 'momentum'  # Chosen from [sgd, momentum, adam, rmsprop]
save_optimizer = True  # Whether to save the optimizer parameters into the checkpoint file.
learning_rate_base = 5e-3
learning_rate_base_batch_size = 64
learning_rate_init = learning_rate_base * ((batch_size * num_gpus) / learning_rate_base_batch_size)
lr_type = 'piecewise'  # Chosen from [fixed, exponential, cosine_decay, cosine_decay_restart, piecewise]
lr_decay_epoch = 5  # Epochs after which learning rate decays. Int or float. Used when chosen `exponential` and `cosine_decay_restart` lr_type.
lr_decay_factor = 0.96  # The learning rate decay factor. Used when chosen `exponential` lr_type.
lr_lower_bound = 1e-6  # The minimum learning rate.
# only used in piecewise lr type
pw_boundaries = [80, 90]  # epoch based boundaries
pw_values = [learning_rate_init, learning_rate_init * 0.1, learning_rate_init * 0.01]

### Load and finetune
# Choose the parts you want to restore the weights. List form.
# restore_include: None, restore_exclude: None  => restore the whole model
# restore_include: None, restore_exclude: scope  => restore the whole model except `scope`
# restore_include: scope1, restore_exclude: scope2  => if scope1 contains scope2, restore scope1 and not restore scope2 (scope1 - scope2)
# choise 1: only restore the darknet body
# restore_include = ['yolov3/darknet53_body']
restore_exclude = None
# choise 2: restore all layers except the last 3 conv2d layers in 3 scale
restore_include = None
# restore_exclude = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6', 'yolov3/yolov3_head/Conv_22']
# restore_exclude = None
# Choose the parts you want to finetune. List form.
# Set to None to train the whole model.
# update_part = ['yolov3/yolov3_head']
update_part = None

### other training strategies
multi_scale_train = False  # Whether to apply multi-scale training strategy. Image size varies from [320, 320] to [640, 640] by default.
use_label_smooth = False # Whether to use class label smoothing strategy.
use_focal_loss = False  # Whether to apply focal loss on the conf loss.
use_mix_up = False  # Whether to use mix up data augmentation strategy.
use_warm_up = True  # whether to use warm up strategy to prevent from gradient exploding.
warm_up_epoch = min(total_epoches * 0.1, 3)  # Warm up training epoches. Set to a larger value if gradient explodes.

### some constants in validation
# nms
nms_threshold = 0.5  # iou threshold in nms operation
score_threshold = 0.001  # threshold of the probability of the classes in nms operation, i.e. score = pred_confs * pred_probs. set lower for higher recall.
nms_topk = 100  # keep at most nms_topk outputs after nms
# mAP eval
eval_threshold = 0.5  # the iou threshold applied in mAP evaluation
use_voc_07_metric = False  # whether to use voc 2007 evaluation metric, i.e. the 11-point metric

### parse some params
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
class_num = len(classes)
train_img_cnt = len(open(train_file, 'r').readlines())
val_img_cnt = len(open(val_file, 'r').readlines())
train_batch_num = int(float(train_img_cnt) / batch_size / num_gpus)

lr_decay_freq = int(train_batch_num * lr_decay_epoch)
pw_boundaries = [float(i) * train_batch_num + global_step for i in pw_boundaries]
