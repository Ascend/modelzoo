# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Config template to train Retinanet."""

# pylint: disable=line-too-long

# For ResNet-50, this freezes the variables of the first conv1 and conv2_x
# layers [1], which leads to higher training speed and slightly better testing
# accuracy. The intuition is that the low-level architecture (e.g., ResNet-50)
# is able to capture low-level features such as edges; therefore, it does not
# need to be fine-tuned for the detection task.
# Note that we need to trailing `/` to avoid the incorrect match.
# [1]: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L198
RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+)\/(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'
REGULARIZATION_VAR_REGEX = r'.*(kernel|weight):0$'

# pylint: disable=line-too-long
RETINANET_CFG = {
    'type': 'retinanet',
    'model_dir': '',
    'use_tpu': True,
    'strategy_type': 'tpu',
    'train': {
        'batch_size': 64,
        'iterations_per_loop': 500,
        'total_steps': 22500,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
            'nesterov': True,  # `False` is better for TPU v3-128.
        },
        'learning_rate': {
            'type': 'step',
            'warmup_learning_rate': 0.0067,
            'warmup_steps': 500,
            'init_learning_rate': 0.08,
            'learning_rate_levels': [0.008, 0.0008],
            'learning_rate_steps': [15000, 20000],
        },
        'checkpoint': {
            'path': '',
            'prefix': '',
        },
        'frozen_variable_prefix': RESNET_FROZEN_VAR_PREFIX,
        'train_file_pattern': '',
        # TODO(b/142174042): Support transpose_input option.
        'transpose_input': False,
        'regularization_variable_regex': REGULARIZATION_VAR_REGEX,
        'l2_weight_decay': 0.0001,
        'input_sharding': False,
    },
    'eval': {
        'batch_size': 8,
        'min_eval_interval': 180,
        'eval_timeout': None,
        'eval_samples': 5000,
        'type': 'box',
        'val_json_file': '',
        'eval_file_pattern': '',
        'input_sharding': True,
        # When visualizing images, set evaluation batch size to 40 to avoid
        # potential OOM.
        'num_images_to_visualize': 0,
    },
    'predict': {
        'predict_batch_size': 8,
    },
    'architecture': {
        'parser': 'retinanet_parser',
        'backbone': 'resnet',
        'multilevel_features': 'fpn',
        'use_bfloat16': False,
    },
    'anchor': {
        'min_level': 3,
        'max_level': 7,
        'num_scales': 3,
        'aspect_ratios': [1.0, 2.0, 0.5],
        'anchor_size': 4.0,
    },
    'retinanet_parser': {
        'use_bfloat16': False,
        'output_size': [640, 640],
        'num_channels': 3,
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 1.0,
        'aug_scale_max': 1.0,
        'use_autoaugment': False,
        'autoaugment_policy_name': 'v0',
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
    },
    'resnet': {
        'resnet_depth': 50,
        'batch_norm': {
            'batch_norm_momentum': 0.997,
            'batch_norm_epsilon': 1e-4,
            'batch_norm_trainable': True,
        },
    },
    'fpn': {
        'min_level': 3,
        'max_level': 7,
        'fpn_feat_dims': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
        'batch_norm': {
            'batch_norm_momentum': 0.997,
            'batch_norm_epsilon': 1e-4,
            'batch_norm_trainable': True,
        },
    },
    'retinanet_head': {
        'min_level': 3,
        'max_level': 7,
        # Note that `num_classes` is the total number of classes including
        # one background classes whose index is 0.
        'num_classes': 91,
        'anchors_per_location': 9,
        'retinanet_head_num_convs': 4,
        'retinanet_head_num_filters': 256,
        'use_separable_conv': False,
        'batch_norm': {
            'batch_norm_momentum': 0.997,
            'batch_norm_epsilon': 1e-4,
            'batch_norm_trainable': True,
        },
    },
    'retinanet_loss': {
        'num_classes': 91,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.1,
        'box_loss_weight': 50,
    },
    'postprocess': {
        'use_batched_nms': False,
        'min_level': 3,
        'max_level': 7,
        'max_total_size': 100,
        'nms_iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pre_nms_num_boxes': 5000,
    },
    'enable_summary': False,
}

RETINANET_RESTRICTIONS = [
    'architecture.use_bfloat16 == retinanet_parser.use_bfloat16',
    'anchor.min_level == retinanet_head.min_level',
    'anchor.max_level == retinanet_head.max_level',
    'anchor.min_level == postprocess.min_level',
    'anchor.max_level == postprocess.max_level',
    'retinanet_head.num_classes == retinanet_loss.num_classes',
]

# pylint: enable=line-too-long
