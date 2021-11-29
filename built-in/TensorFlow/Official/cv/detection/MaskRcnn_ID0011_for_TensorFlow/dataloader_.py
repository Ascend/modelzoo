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
#

# ==============================================================================
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
import tensorflow.compat.v1 as tf

import anchors
import coco_utils
import preprocess_ops
import spatial_transform_ops
from object_detection import tf_example_decoder

from utils import box_utils
from utils import dataloader_utils
from utils import input_utils
from dataloader import extract_objects_parser

MAX_NUM_INSTANCES = 100
MAX_NUM_VERTICES_PER_INSTANCE = 1500
MAX_NUM_POLYGON_LIST_LEN = 2 * MAX_NUM_VERTICES_PER_INSTANCE * MAX_NUM_INSTANCES
POLYGON_PAD_VALUE = coco_utils.POLYGON_PAD_VALUE


def _prepare_labels_for_eval(data,
                             target_num_instances=MAX_NUM_INSTANCES,
                             target_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN,
                             use_instance_mask=False):
  """Create labels dict for infeed from data of tf.Example."""
  image = data['image']
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]
  boxes = data['groundtruth_boxes']
  classes = data['groundtruth_classes']
  classes = tf.cast(classes, dtype=tf.float32)
  num_labels = tf.shape(classes)[0]
  boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [target_num_instances, 4])
  classes = preprocess_ops.pad_to_fixed_size(classes, -1,
                                             [target_num_instances, 1])
  is_crowd = data['groundtruth_is_crowd']
  is_crowd = tf.cast(is_crowd, dtype=tf.float32)
  is_crowd = preprocess_ops.pad_to_fixed_size(is_crowd, 0,
                                              [target_num_instances, 1])
  labels = {}
  labels['width'] = width
  labels['height'] = height
  labels['groundtruth_boxes'] = boxes
  labels['groundtruth_classes'] = classes
  labels['num_groundtruth_labels'] = num_labels
  labels['groundtruth_is_crowd'] = is_crowd

  if use_instance_mask:
    polygons = data['groundtruth_polygons']
    polygons = preprocess_ops.pad_to_fixed_size(polygons, POLYGON_PAD_VALUE,
                                                [target_polygon_list_len, 1])
    labels['groundtruth_polygons'] = polygons
    if 'groundtruth_area' in data:
      groundtruth_area = data['groundtruth_area']
      groundtruth_area = preprocess_ops.pad_to_fixed_size(
          groundtruth_area, 0, [target_num_instances, 1])
      labels['groundtruth_area'] = groundtruth_area

  return labels


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self,
               file_pattern,
               mode=tf.estimator.ModeKeys.TRAIN,
               num_examples=0,
               use_fake_data=False,
               use_instance_mask=False,
               max_num_instances=MAX_NUM_INSTANCES,
               max_num_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN):
    self._file_pattern = file_pattern
    self._max_num_instances = max_num_instances
    self._max_num_polygon_list_len = max_num_polygon_list_len
    self._mode = mode
    self._num_examples = num_examples
    self._use_fake_data = use_fake_data
    self._use_instance_mask = use_instance_mask

    self._include_mask = True
    self._skip_crowd_during_training = True
    self._aug_rand_hflip = True

    self._min_level = 2
    self._max_level = 6
    self._aug_scale_min = 0.5
    self._aug_scale_max = 2
    self._output_size = [1024, 1024]
    self._mask_crop_size = 112

    self._copy_paste_occluded_obj_threshold = 300
    self._copy_paste_box_update_threshold = 10

  def _transform_mask(self, image_shape, scale, offset, mask):
    """Transform input mask according to the image info (scale, offset)"""
    image_scaled_shape = tf.round(
      tf.cast(image_shape, tf.float32) * scale
    )
    image_scaled_shape = tf.cast(image_scaled_shape, tf.int32)

    offset = tf.cast(offset, tf.int32)
    mask_shape = tf.shape(mask)
    mask = tf.image.pad_to_bounding_box(
      mask, offset[0], offset[1],
      tf.maximum(image_scaled_shape[0], mask_shape[0]) + offset[0],
      tf.maximum(image_scaled_shape[1], mask_shape[1]) + offset[1],
    )
    mask = mask[0:image_scaled_shape[0], 0:image_scaled_shape[1], :]
    mask = tf.image.resize(mask, image_shape)
    return mask

  def _get_occluded_bbox(self, updated_bbox, bbox):
    # finds bbox coordinates which are occluded by the new pasted objects.
    # if the difference between the bounding box coordinates of updated masks
    # and the original bounding box are larger than a threshold then those 
    # coordinates are considered as occluded
    return tf.greater(tf.abs(updated_bbox - tf.cast(bbox, bbox.dtype)),
    self._copy_paste_box_update_threshold)

  def _get_visible_masks_indices(self, masks, boxes_, cropped_boxes):
    """return indices of not fully occluded objects"""
    occluded_objects = tf.reduce_any(
      self._get_occluded_bbox(boxes_, cropped_boxes)
    )
    areas = tf.reduce_sum(masks, axis=[1, 2])
    # among the occluded objects, find the objects that their mask area is 
    # less than copy_paste_occluded_obj_threshold.These objects are considered
    # as fully occluded objects and will be removed from the ground truth
    indices = tf.where(
      tf.math.logical_or(
        tf.greater(areas, self._copy_paste_occluded_obj_threshold),
        tf.math.logical_not(occluded_objects)
      )
    )
    indices = tf.reshape(indices, [-1])
    return indices

  def _compute_boxes_using_masks(self, masks, image_shape, image_info, image_scale, offset):
    """computes bounding boxes using masks"""
    masks = tf.cast(masks, tf.int8)
    x = tf.reduce_max(masks, axis=1)
    xmin = tf.cast(tf.argmax(x, 1), tf.int16)
    xmax = tf.cast(image_shape[1], tf.int16) - tf.cast(tf.argmax(tf.reverse(x, [1]), 1), tf.int16)
    y = tf.reduce_max(masks, axis=2)
    ymin = tf.cast(tf.argmax(y, 1), tf.int16)
    ymax = tf.cast(image_shape[0], tf.int16) - tf.cast(tf.argmax(tf.reverse(y, [1]), 1), tf.int16)
    bbox = tf.stack([ymin, xmin, ymax, xmax], -1)

    # clips boxes
    bbox = tf.cast(bbox, tf.float32)
    bbox = input_utils.resize_and_crop_boxes(
      bbox, image_scale, image_info[1, :], offset
    )
    bbox += tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    bbox /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])

    return bbox

  def _parse_train_data_extractObjs(self, data):
    """
    parses data for training.
    Args:
      data: the decoded tensor dictionary from tfexampledecoder

    returns:
      image: image tensor that is preprocessed to have normalized value and dimension [output_size[0], output_size[1], 4]
      labels: a dictionary of tensors used for training.
    """
    classes = data['groundtruth_classes2']
    boxes = data['groundtruth_boxes2']
    if self._include_mask:
      masks = data['groundtruth_instance_masks2']

    is_crowds = data['groundtruth_is_crowd2']
    # skips annotation with 'is_crowd' = True
    if self._skip_crowd_during_training:
      num_groundtrtuhs = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
          tf.greater(tf.size(data['groundtruth_is_crowd2']), 0),
          lambda: data['groundtruth_is_crowd2'],
          lambda: tf.zeros_like(data['groundtruth_classes2'], dtype=tf.bool)
        )
      indices = tf.where(tf.logical_not(indices))
      classes = tf.gather_nd(classes, indices)
      boxes = tf.gather_nd(boxes, indices)
      if self._include_mask:
        masks = tf.gather_nd(masks, indices)
    
    # gets original image and its size
    image = data['image2']
    image_shape = tf.shape(image)[0:2]

    # normalizes image with mean and std pixel values
    image = input_utils.normalize_image(image)

    # flips image randomly during training
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = input_utils.random_horizontal_flip(
          image, boxes, masks
        )
      else:
        image, boxes = input_utils.random_horizontal_flip(
          image, boxes
        )
    
    # converts boxes from normaliezd coordinates to pixel coordinates.
    # now the coordinates of boxes are w.r.t. the original image.
    boxes = box_utils.denormalize_boxes(boxes, image_shape)

    # resizes and crops image
    image, image_info, _ = input_utils.resize_and_crop_image(
      image,
      self._output_size,
      padded_size=input_utils.compute_padded_size(
        self._output_size, 2 ** self._max_level
      ),
      aug_scale_min=self._aug_scale_min,
      aug_scale_max=self._aug_scale_max
    )

    # resizes and crops boxes
    # now the coordinates of boxes are w.r.t. the scaled image.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = input_utils.resize_and_crop_boxes(
      boxes, image_scale, image_info[1, :], offset
    )

    # filters out groundtruth boxes that are all zeros
    indices = box_utils.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    if self._include_mask:
      masks = tf.gather(masks, indices)
      uncropped_masks = tf.cast(masks, tf.int8)
      uncropped_masks = tf.expand_dims(uncropped_masks, axis=3)
      uncropped_masks = input_utils.resize_and_crop_masks(
        uncropped_masks, image_scale, self._output_size, offset
      )

      # transfer boxes to the original image space and do normalization
      cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
      cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
      cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
        tf.expand_dims(masks, axis=-1),
        cropped_boxes,
        box_indices = tf.range(num_masks, dtype=tf.int32),
        crop_size = [self._mask_crop_size, self._mask_crop_size],
        method='bilinear'
      )
      masks = tf.squeeze(masks, axis=-1)
    indices = tf.range(start=0, limit=tf.shape(classes)[0], dtype=tf.int32)

    # samples the numbers of masks for pasting
    m = tf.random.uniform(shape=[], maxval=tf.shape(classes)[0]+1, dtype=tf.int32)
    m = tf.math.minimum(m, tf.shape(classes)[0])

    # shuffle the indices of objects and keep the first m objects for pasting
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_indices = tf.slice(shuffled_indices, [0], [m])
    boxes = tf.gather(boxes, shuffled_indices)
    masks = tf.gather(masks, shuffled_indices)
    classes = tf.gather(classes, shuffled_indices)
    uncropped_masks = tf.gather(uncropped_masks, shuffled_indices)
    pasted_objects_mask = tf.reduce_max(uncropped_masks, 0)
    pasted_objects_mask = tf.cast(pasted_objects_mask, tf.bool)
    labels = {
      'image':image,
      'image_info':image_info,
      'num_groundtrtuhs':tf.shape(classes)[0],
      'boxes':boxes,
      'masks':masks,
      'classes':classes,
      'pasted_objects_mask':pasted_objects_mask,
    }
    return labels

  def _create_dataset_fn(self):
    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    return _prefetch_dataset

  def _create_example_decoder(self):
    return tf_example_decoder.TfExampleDecoder(
        use_instance_mask=self._use_instance_mask)

  def _create_dataset_parser_fn(self, params):
    """Create parser for parsing input data (dictionary)."""
    example_decoder = self._create_example_decoder()

    def _dataset_parser(value, value2=None):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        features: a dictionary that contains the image and auxiliary
          information. The following describes {key: value} pairs in the
          dictionary.
          image: Image tensor that is preproessed to have normalized value and
            fixed dimension [image_size, image_size, 3]
          image_info: image information that includes the original height and
            width, the scale of the proccessed image to the original image, and
            the scaled height and width.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
        labels: a dictionary that contains auxiliary information plus (optional)
          labels. The following describes {key: value} pairs in the dictionary.
          `labels` is only for training.
          score_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors]. The height_l and width_l
            represent the dimension of objectiveness score at l-th level.
          box_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
          gt_boxes: Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
             fixed dimension [self._max_num_instances, 4].
          gt_classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          cropped_gt_masks: groundtrugh masks cropped by the bounding box and
            resized to a fixed size determined by params['gt_mask_size']
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        # extract data2 objs here
        if value2 is None:
          data2 = self._parse_train_data_extractObjs(data)
        else:
          data2 = value2

        data['groundtruth_is_crowd'] = tf.cond(
            tf.greater(tf.size(data['groundtruth_is_crowd']), 0),
            lambda: data['groundtruth_is_crowd'],
            lambda: tf.zeros_like(data['groundtruth_classes'], dtype=tf.bool))
        image = data['image']
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        orig_image = image
        source_id = data['source_id']
        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        if (self._mode == tf.estimator.ModeKeys.PREDICT or
            self._mode == tf.estimator.ModeKeys.EVAL):
          image = preprocess_ops.normalize_image(image)
          if params['resize_method'] == 'retinanet':
            image, image_info, _, _, _ = preprocess_ops.resize_crop_pad(
                image, params['image_size'], 2 ** params['max_level'])
          else:
            image, image_info, _, _, _ = preprocess_ops.resize_crop_pad_v2(
                image, params['short_side'], params['long_side'],
                2 ** params['max_level'])
          if params['precision'] == 'bfloat16':
            image = tf.cast(image, dtype=tf.bfloat16)

          features = {
              'images': image,
              'image_info': image_info,
              'source_ids': source_id,
          }
          if params['visualize_images_summary']:
            resized_image = tf.image.resize_images(orig_image,
                                                   params['image_size'])
            features['orig_images'] = resized_image
          if (params['include_groundtruth_in_features'] or
              self._mode == tf.estimator.ModeKeys.EVAL):
            labels = _prepare_labels_for_eval(
                data,
                target_num_instances=self._max_num_instances,
                target_polygon_list_len=self._max_num_polygon_list_len,
                use_instance_mask=params['include_mask'])
            return {'features': features, 'labels': labels}
          else:
            return {'features': features}

        elif self._mode == tf.estimator.ModeKeys.TRAIN:
          instance_masks = None
          if self._use_instance_mask:
            instance_masks = data['groundtruth_instance_masks']
          boxes = data['groundtruth_boxes']
          classes = data['groundtruth_classes']
          classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
          if not params['use_category']:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

          if (params['skip_crowd_during_training'] and
              self._mode == tf.estimator.ModeKeys.TRAIN):
            indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
            classes = tf.gather_nd(classes, indices)
            boxes = tf.gather_nd(boxes, indices)
            if self._use_instance_mask:
              instance_masks = tf.gather_nd(instance_masks, indices)

          image = preprocess_ops.normalize_image(image)
          if params['input_rand_hflip']:
            flipped_results = (
                preprocess_ops.random_horizontal_flip(
                    image, boxes=boxes, masks=instance_masks))
            if self._use_instance_mask:
              image, boxes, instance_masks = flipped_results
            else:
              image, boxes = flipped_results
          # Scaling, jittering and padding.
          if params['resize_method'] == 'retinanet':
            image_shape = tf.shape(image)[0:2]
            boxes = box_utils.denormalize_boxes(boxes, image_shape)
            image, image_info_copyPaste, image_info = input_utils.resize_and_crop_image(
              image,
              params['image_size'],
              padded_size=input_utils.compute_padded_size(
                params['image_size'], 2 ** self._max_level
              ),
              aug_scale_min=params['aug_scale_min'],
              aug_scale_max=params['aug_scale_max']
            )

            # resizes and crops boxes
            # now the coordinates of boxes are w.r.t. the scaled image
            image_scale = image_info_copyPaste[2, :]
            offset = image_info_copyPaste[3, :]
            boxes = input_utils.resize_and_crop_boxes(
              boxes, image_scale, image_info_copyPaste[1, :], offset
            )
            indices = box_utils.get_non_empty_box_indices(boxes)
            boxes = tf.gather(boxes, indices)
            classes = tf.gather(classes, indices)
          else:
            image, image_info, boxes, classes, cropped_gt_masks = (
                preprocess_ops.resize_crop_pad_v2(
                    image,
                    params['short_side'],
                    params['long_side'],
                    2 ** params['max_level'],
                    aug_scale_min=params['aug_scale_min'],
                    aug_scale_max=params['aug_scale_max'],
                    boxes=boxes,
                    classes=classes,
                    masks=instance_masks,
                    crop_mask_size=params['gt_mask_size']))

          data2['classes'] = tf.cast(data2['classes'], dtype=tf.float32)

          instance_masks_init = tf.identity(instance_masks)
          indices_init = tf.identity(indices)
          boxes_init = tf.identity(boxes)
          classes_init = tf.identity(classes)
          image_init = tf.identity(image)

          _copy_paste_aug = True
          if _copy_paste_aug:
            # paste objects and creates a new composed image
            compose_mask = tf.cast(data2['pasted_objects_mask'],image.dtype) * tf.ones_like(image)

            image = image * (1 - compose_mask) + data2['image'] * compose_mask

          if self._include_mask:
            masks = tf.gather(instance_masks, indices)
            if _copy_paste_aug:
              pasted_objects_mask = self._transform_mask(
                image_shape, image_scale, offset,
                tf.cast(data2['pasted_objects_mask'], tf.int8)
              )
              pasted_objects_mask = tf.cast(pasted_objects_mask, tf.int8)
              pasted_objects_mask = tf.expand_dims(
                tf.squeeze(pasted_objects_mask, -1), 0) * tf.ones(tf.shape(masks), dtype=pasted_objects_mask.dtype)
              masks = tf.where(
                tf.equal(pasted_objects_mask, 1), tf.zeros_like(masks), masks
              )
            cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
            cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])

            if _copy_paste_aug:
              # computes bounding boxes of objects using updated masks
              boxes_ = self._compute_boxes_using_masks(
                masks, image_shape, image_info_copyPaste, image_scale, offset
              )
              # filters out objects that are fully occluded in the new image
              indices = self._get_visible_masks_indices(
                masks, boxes_, cropped_boxes
              )
              boxes_ = tf.gather(boxes_, indices)
              boxes = tf.gather(boxes, indices)
              cropped_boxes = tf.gather(cropped_boxes, indices)
              masks = tf.gather(masks, indices)
              classes = tf.gather(classes, indices)

              # update bounding boxes of which are occluded by new pasted objects
              def update_bboxes(boxes_, cropped_boxes):
                occluded_bbox = self._get_occluded_bbox(boxes_, cropped_boxes)
                cropped_boxes = tf.where(
                  occluded_bbox,
                  tf.cast(boxes_, cropped_boxes.dtype),
                  cropped_boxes
                )
                boxes = input_utils.resize_and_crop_boxes(
                  cropped_boxes, image_scale, image_info_copyPaste[1, :], offset
                )
                return boxes, cropped_boxes
              boxes, cropped_boxes = update_bboxes(boxes_, cropped_boxes)
            cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
            num_masks = tf.shape(masks)[0]
            masks = tf.image.crop_and_resize(
              tf.expand_dims(masks, axis=-1),
              cropped_boxes,
              box_indices=tf.range(num_masks, dtype=tf.int32),
              crop_size=[self._mask_crop_size, self._mask_crop_size],
              method='bilinear'
            )
            masks = tf.squeeze(masks, axis=-1)
            cropped_gt_masks = masks
          else:
            cropped_gt_masks = None
          
          if _copy_paste_aug:
            if self._include_mask:
              masks = tf.concat([masks, data2['masks']], axis=0)
            data2['classes'] = tf.reshape(tf.cast(data2['classes'], dtype=tf.float32), [-1, 1])
            boxes = tf.concat([boxes, data2['boxes']], axis=0)
            classes = tf.concat([classes, data2['classes']], axis=0)

          if cropped_gt_masks is not None:
            cropped_gt_masks = tf.pad(
                cropped_gt_masks,
                paddings=tf.constant([[0, 0,], [2, 2,], [2, 2]]),
                mode='CONSTANT',
                constant_values=0.)

          padded_height, padded_width, _ = image.get_shape().as_list()
          padded_image_size = (padded_height, padded_width)
          input_anchors = anchors.Anchors(
              params['min_level'],
              params['max_level'],
              params['num_scales'],
              params['aspect_ratios'],
              params['anchor_scale'],
              padded_image_size)
          anchor_labeler = anchors.AnchorLabeler(
              input_anchors,
              params['num_classes'],
              params['rpn_positive_overlap'],
              params['rpn_negative_overlap'],
              params['rpn_batch_size_per_im'],
              params['rpn_fg_fraction'])

          def no_aug(instance_masks, indices, boxes, classes, image):
            masks = tf.gather(instance_masks, indices)

            cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
            cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])

            cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
            num_masks = tf.shape(masks)[0]
            masks = tf.image.crop_and_resize(
              tf.expand_dims(masks, axis=-1),
              cropped_boxes,
              box_indices=tf.range(num_masks, dtype=tf.int32),
              crop_size=[self._mask_crop_size, self._mask_crop_size],
              method='bilinear'
            )
            masks = tf.squeeze(masks, axis=-1)
            cropped_gt_masks = masks

            cropped_gt_masks = tf.pad(
              cropped_gt_masks,
              paddings=tf.constant([[0, 0], [2, 2], [2, 2]]),
              mode='CONSTANT',
              constant_values=0.
            )
            return cropped_gt_masks, boxes, classes, image

          if tf.shape(classes)[0] > 92:
            cropped_gt_masks, boxes, classes, image = no_aug(instance_masks_init, indices_init, boxes_init, classes_init, image_init)

          # Assign anchors.
          score_targets, box_targets = anchor_labeler.label_anchors(
              boxes, classes)

          # Pad groundtruth data.
          boxes = preprocess_ops.pad_to_fixed_size(
              boxes, -1, [self._max_num_instances, 4])
          classes = preprocess_ops.pad_to_fixed_size(
              classes, -1, [self._max_num_instances, 1])

          # Pads cropped_gt_masks.
          if self._use_instance_mask:
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks, tf.stack([tf.shape(cropped_gt_masks)[0], -1]))
            cropped_gt_masks = preprocess_ops.pad_to_fixed_size(
                cropped_gt_masks, -1,
                [self._max_num_instances, (params['gt_mask_size'] + 4) ** 2])
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks,
                [self._max_num_instances, params['gt_mask_size'] + 4,
                 params['gt_mask_size'] + 4])

          if params['precision'] == 'bfloat16':
            image = tf.cast(image, dtype=tf.bfloat16)

          features = {
              'images': image,
              'image_info': image_info,
              'source_ids': source_id,
          }
          labels = {}
          for level in range(params['min_level'], params['max_level'] + 1):
            labels['score_targets_%d' % level] = score_targets[level]
            labels['box_targets_%d' % level] = box_targets[level]
          labels['gt_boxes'] = boxes
          labels['gt_classes'] = classes
          if self._use_instance_mask:
            labels['cropped_gt_masks'] = cropped_gt_masks
          return features, labels

    return _dataset_parser

  def get_data(self, _file_pattern, dataset_fn, input_context=None):
    
    dataset = tf.data.Dataset.list_files(
        _file_pattern, shuffle=(self._mode == tf.estimator.ModeKeys.TRAIN), seed=0)
    if input_context is not None:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            dataset_fn,
            cycle_length=32,
            sloppy=(self._mode == tf.estimator.ModeKeys.TRAIN)))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(64, seed=0)
    return dataset

  def _create_dataset_parser_fn_pre(self, params=None):
    parse_pre = extract_objects_parser.Parser(
      [1024, 1024],
      params['min_level'],
      params['max_level'],
      aug_rand_hflip=True,
      aug_scale_min=0.1,
      aug_scale_max=2.0,
      skip_crowd_during_training=True,
      include_mask=True,
      mask_crop_size=112
    )
    return parse_pre

  def __call__(self, params, input_context=None):
    dataset_parser_fn = self._create_dataset_parser_fn(params)
    dataset_fn = self._create_dataset_fn()
    batch_size = params['batch_size'] if 'batch_size' in params else 1
    dataset = self.get_data(self._file_pattern, dataset_fn, input_context)
    dataset_p = self.get_data(self._file_pattern, dataset_fn, input_context)
    pre_parser_fn = self._create_dataset_parser_fn_pre(params)
    dataset_p = dataset_p.map(
      pre_parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset_p = dataset_p.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_p = dataset_p.filter(
      lambda data:tf.greater(data['num_groundtrtuhs'], 0)
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((dataset, dataset_p))
    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(
            dataset_parser_fn,
            num_parallel_calls=256)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Enable TPU performance optimization: transpose input, space-to-depth
    # image transform, or both.
    if (self._mode == tf.estimator.ModeKeys.TRAIN and
        (params['transpose_input'] or
         (params['backbone'].startswith('resnet') and
          params['conv0_space_to_depth_block_size'] > 0))):

      def _transform_images(features, labels):
        """Transforms images."""
        images = features['images']
        if (params['backbone'].startswith('resnet') and
            params['conv0_space_to_depth_block_size'] > 0):
          # Transforms images for TPU performance.
          features['images'] = (
              spatial_transform_ops.fused_transpose_and_space_to_depth(
                  images,
                  params['conv0_space_to_depth_block_size'],
                  params['transpose_input']))
        else:
          features['images'] = tf.transpose(features['images'], [1, 2, 3, 0])
        return features, labels

      dataset = dataset.map(_transform_images, num_parallel_calls=256)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self._num_examples > 0:
      dataset = dataset.take(self._num_examples)
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()
    return dataset
