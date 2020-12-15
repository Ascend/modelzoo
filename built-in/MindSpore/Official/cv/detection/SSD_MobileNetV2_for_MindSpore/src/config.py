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
#" ============================================================================

"""Config parameters for SSD models."""

from easydict import EasyDict as ed

config = ed({
    "img_shape": [300, 300],
    "num_ssd_boxes": 1917,
    "neg_pre_positive": 3,
    "match_thershold": 0.5,
    "nms_thershold": 0.6,
    "min_score": 0.1,
    "max_boxes": 100,

    # learing rate settings
    "global_step": 0,
    "lr_init": 0.001,
    "lr_end_rate": 0.001,
    "warmup_epochs": 2,
    "momentum": 0.9,
    "weight_decay": 1.5e-4,

    # network
    "num_default": [3, 6, 6, 6, 6, 6],
    "extras_in_channels": [256, 576, 1280, 512, 256, 256],
    "extras_out_channels": [576, 1280, 512, 256, 256, 128],
    "extras_srides": [1, 1, 2, 2, 2, 2],
    "extras_ratio": [0.2, 0.2, 0.2, 0.25, 0.5, 0.25],
    "feature_size": [19, 10, 5, 3, 2, 1],
    "min_scale": 0.2,
    "max_scale": 0.95,
    "aspect_ratios": [(2,), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)],
    "steps": (16, 32, 64, 100, 150, 300),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.75,

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    "mindrecord_dir": "/data/MindRecord_COCO",
    "coco_root": "/data/coco2017",
    "train_data_type": "train2017",
    "val_data_type": "val2017",
    "instances_set": "annotations/instances_{}.json",
    "coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush'),
    "num_classes": 81,
    # The annotation.json position of voc validation dataset.
    "voc_root": "",
    # voc original dataset.
    "voc_dir": "",
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    "image_dir": "",
    "anno_path": "",
})
