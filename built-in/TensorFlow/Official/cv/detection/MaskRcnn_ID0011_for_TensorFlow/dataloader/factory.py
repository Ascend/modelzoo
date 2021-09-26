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
"""Data loader factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataloader import classification_parser
from dataloader import extract_objects_parser
from dataloader import maskrcnn_parser
from dataloader import maskrcnn_parser_with_copy_paste
from dataloader import mode_keys as ModeKeys
from dataloader import retinanet_parser
from dataloader import segmentation_parser
from dataloader import shapemask_parser


def parser_generator(params, mode):
  """Generator function for various dataset parser."""
  if params.architecture.parser == 'classification_parser':
    parser_params = params.classification_parser
    parser_fn = classification_parser.Parser(
        output_size=parser_params.output_size,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode)
  elif params.architecture.parser == 'retinanet_parser':
    anchor_params = params.anchor
    parser_params = params.retinanet_parser
    parser_fn = retinanet_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        match_threshold=parser_params.match_threshold,
        unmatched_threshold=parser_params.unmatched_threshold,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        aug_policy=parser_params.aug_policy,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode,
        regenerate_source_id=parser_params.regenerate_source_id)
  elif params.architecture.parser == 'maskrcnn_parser':
    anchor_params = params.anchor
    parser_params = params.maskrcnn_parser
    parser_fn = maskrcnn_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        rpn_match_threshold=parser_params.rpn_match_threshold,
        rpn_unmatched_threshold=parser_params.rpn_unmatched_threshold,
        rpn_batch_size_per_im=parser_params.rpn_batch_size_per_im,
        rpn_fg_fraction=parser_params.rpn_fg_fraction,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        include_mask=params.architecture.include_mask,
        mask_crop_size=parser_params.mask_crop_size,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode)
    if mode == ModeKeys.TRAIN and parser_params.copy_paste:
      parser_fn = maskrcnn_parser_with_copy_paste.Parser(
          output_size=parser_params.output_size,
          min_level=params.architecture.min_level,
          max_level=params.architecture.max_level,
          num_scales=anchor_params.num_scales,
          aspect_ratios=anchor_params.aspect_ratios,
          anchor_size=anchor_params.anchor_size,
          rpn_match_threshold=parser_params.rpn_match_threshold,
          rpn_unmatched_threshold=parser_params.rpn_unmatched_threshold,
          rpn_batch_size_per_im=parser_params.rpn_batch_size_per_im,
          rpn_fg_fraction=parser_params.rpn_fg_fraction,
          aug_rand_hflip=parser_params.aug_rand_hflip,
          aug_scale_min=parser_params.aug_scale_min,
          aug_scale_max=parser_params.aug_scale_max,
          skip_crowd_during_training=parser_params.skip_crowd_during_training,
          max_num_instances=parser_params.max_num_instances,
          include_mask=params.architecture.include_mask,
          mask_crop_size=parser_params.mask_crop_size,
          use_bfloat16=params.architecture.use_bfloat16,
          mode=mode)
  elif params.architecture.parser == 'extract_objects_parser':
    parser_params = params.maskrcnn_parser
    parser_fn = extract_objects_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        include_mask=params.architecture.include_mask,
        mask_crop_size=parser_params.mask_crop_size)
  elif params.architecture.parser == 'shapemask_parser':
    anchor_params = params.anchor
    parser_params = params.shapemask_parser
    parser_fn = shapemask_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        use_category=parser_params.use_category,
        outer_box_scale=parser_params.outer_box_scale,
        box_jitter_scale=parser_params.box_jitter_scale,
        num_sampled_masks=parser_params.num_sampled_masks,
        mask_crop_size=parser_params.mask_crop_size,
        mask_min_level=parser_params.mask_min_level,
        mask_max_level=parser_params.mask_max_level,
        upsample_factor=parser_params.upsample_factor,
        match_threshold=parser_params.match_threshold,
        unmatched_threshold=parser_params.unmatched_threshold,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        use_bfloat16=params.architecture.use_bfloat16,
        mask_train_class=parser_params.mask_train_class,
        mode=mode)
  elif params.architecture.parser == 'segmentation_parser':
    parser_params = params.segmentation_parser
    parser_fn = segmentation_parser.Parser(
        output_size=parser_params.output_size,
        resize_eval=parser_params.resize_eval,
        ignore_label=parser_params.ignore_label,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        aug_policy=parser_params.aug_policy,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode)
  else:
    raise ValueError('Parser %s is not supported.' % params.architecture.parser)

  return parser_fn
