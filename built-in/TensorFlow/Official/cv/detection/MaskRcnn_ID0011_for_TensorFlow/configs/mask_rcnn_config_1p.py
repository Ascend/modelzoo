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
#

# ==============================================================================
"""Config to train Mask RCNN."""

MASK_RCNN_CFG = {
    # runtime parameters
    'transpose_input': True,
    'iterations_per_loop': 30000,
    'num_cores': 8,
    'use_tpu': False,
    # input preprocessing parameters
    'resize_method': 'retinanet',  # or 'faster_rcnn'
    'image_size': [1024, 1024], #[768, 1280], #
    'short_side': 800,
    'long_side': 1333,
    'input_rand_hflip': True,
    'aug_scale_min': 1.0,
    'aug_scale_max': 1.0,
    'gt_mask_size': 112,
    # dataset specific parameters
    'num_classes': 91,
    'skip_crowd_during_training': True,
    'use_category': True,
    # Region Proposal Network
    'rpn_positive_overlap': 0.7,
    'rpn_negative_overlap': 0.3,
    'rpn_batch_size_per_im': 256,
    'rpn_fg_fraction': 0.5,
    'rpn_pre_nms_topn': 2000,
    'rpn_post_nms_topn': 1000,
    'rpn_nms_threshold': 0.7,
    'rpn_min_size': 0.,
    # Proposal layer.
    'batch_size_per_im': 512,
    'fg_fraction': 0.25,
    'fg_thresh': 0.5,
    'bg_thresh_hi': 0.5,
    'bg_thresh_lo': 0.,
    # Faster-RCNN heads.
    'fast_rcnn_mlp_head_dim': 1024,
    # Whether or not to output box features. The box features are commonly
    # used as the visual features for many SOTA image-text generation models,
    # such as image-captioning and VQA.
    'output_box_features': False,
    'bbox_reg_weights': [10., 10., 5., 5.],
    # Mask-RCNN heads.
    'include_mask': False,  # whether or not to include mask branch.
    'mrcnn_resolution': 28,
    # evaluation
    'test_detections_per_image': 100,
    'test_nms': 0.5,
    'test_rpn_pre_nms_topn': 1000,
    'test_rpn_post_nms_topn': 1000,
    'test_rpn_nms_thresh': 0.7,
    # Whether or not tf.combined_non_max_suppression is used. Note that this
    # op is only available on CPU/GPU, and for tf version greater than 1.13.
    'use_batched_nms': True,
    # model architecture
    'min_level': 2,
    'max_level': 6,
    'num_scales': 1,
    'aspect_ratios': [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
    'anchor_scale': 8.0,
    # Number of groups to normalize in the distributed batch normalization.
    # Replicas will evenly split into groups. If positive, use tpu specifc
    # batch norm implemenation which calculates mean and variance accorss all
    # the replicas.
    'num_batch_norm_group': -1,
    # is batchnorm training mode
    'is_training_bn': False,
    # optimization
    'momentum': 0.9,
    # localization loss
    'delta': 0.1,
    'rpn_box_loss_weight': 1.0,
    'fast_rcnn_box_loss_weight': 1.0,
    'mrcnn_weight_loss_mask': 1.0,
    # l2 regularization weight.
    'l2_weight_decay': 1e-4,
    # ---------- Training configurations ----------
    'train_batch_size': 2, #64,
    'learning_rate_type': 'step',  # 'step' or 'cosine'.
    'init_learning_rate': 0.0025, #0.005, #0.02,
    'warmup_learning_rate': 0.00067,
    'warmup_steps': 500,
    'learning_rate_levels': [0.00025, 0.000025], #[0.0005, 0.00005], #[0.002, 0.0002],
    'learning_rate_steps': [480000, 640000], #[180000, 240000], #[15000, 20000],
    'total_steps': 90000, #720000, #360000, #22500,
    'training_file_pattern': '/home/data_new/coco_official_2017/tfrecord/train2017*',
    # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # 'resnet200'] for resnet backbone.
    # ['mnasnet-a1', 'mnasnet-b1', 'mnasnet-small'] for mnasnet backbone.
    'backbone': 'resnet50',
    'checkpoint': '/home/chenxinghao/tf_pretrained/tpu/resnet-50-imagenet/model.ckpt',
    # Optional string filepath to a checkpoint to warm-start from. By default
    # all variables are warm-started, and it is assumed that vocabularies and
    # `tf.Tensor` names are unchanged. One can use the
    # `skip_checkpoint_variables` to skip some variables.
    'warm_start_path': '',
    # TPU performance optimization.
    'precision': 'bfloat16',
    'conv0_kernel_size': 7,
    'conv0_space_to_depth_block_size': 2,
    'use_host_call': False,
    # One of ['momentum', 'adam', 'adadelta', 'adagrad', 'rmsprop', 'lars'].
    'optimizer': 'momentum',
    # Gradient clipping is a fairly coarse heuristic to stabilize training.
    # This model clips the gradient by its L2 norm globally (i.e., across
    # all variables), using a threshold obtained from multiplying this
    # parameter with sqrt(number_of_weights), to have a meaningful value
    # across both training phases and different sizes of imported modules.
    # Refer value: 0.02, for 25M weights, yields clip norm 10.
    # Zero or negative number means no clipping.
    'global_gradient_clip_ratio': -1.0,
    # Skips loading variables from the resnet checkpoint. It is used for
    # skipping nonexistent variables from the constructed graph. The list
    # of loaded variables is constructed from the scope 'resnetX', where 'X'
    # is depth of the resnet model. Supports regular expression.
    'skip_checkpoint_variables': '^NO_SKIP$',
    # Weight decay for LARS optimizer.
    'lars_weight_decay': 1e-4,
    # ---------- Eval configurations ----------
    'eval_batch_size': 1,
    'num_steps_per_eval': 45000, #2500,
    'eval_samples': 5000,
    'validation_file_pattern': '/home/data_new/coco_official_2017/tfrecord/val2017*',
    'val_json_file': '/home/data_new/coco_official_2017/annotations/instances_val2017.json',
    # If `val_json_file` is not provided, one can also read groundtruth
    # from input by setting `include_groundtruth_in_features`:True.
    'include_groundtruth_in_features': False,
    # Visualizes images and detection boxes on TensorBoard.
    'visualize_images_summary': False,
}

MASK_RCNN_RESTRICTIONS = [
]
