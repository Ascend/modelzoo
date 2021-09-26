# Lint as: python2, python3
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
"""Post-processing model outputs to generate detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from six.moves import range
import tensorflow.compat.v1 as tf

from ops import nms
from utils import box_utils


def generate_detections_factory(params):
  """Factory to select function to generate detection."""
  if params.use_batched_nms or params.nms_version == 'batched':
    func = functools.partial(
        _generate_detections_batched,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold)
  elif params.nms_version == 'v2':
    func = functools.partial(
        _generate_detections_v2,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold,
        pre_nms_num_boxes=params.pre_nms_num_boxes)
  else:
    func = functools.partial(
        _generate_detections_v1,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold,
        pre_nms_num_boxes=params.pre_nms_num_boxes)
  return func


def _generate_detections_v1(boxes,
                            scores,
                            max_total_size=100,
                            nms_iou_threshold=0.3,
                            score_threshold=0.05,
                            pre_nms_num_boxes=5000):
  """Generate the final detections given the model outputs.

  This uses batch unrolling, which is TPU compatible.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or
      [batch_size, N, 1, 4], which box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nmsed_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nmsed_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nmsed_classes: `int` Tensor of shape [batch_size, max_total_size]
      representing classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    batch_size = scores.get_shape().as_list()[0]
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    for i in range(batch_size):
      (nmsed_boxes_i, nmsed_scores_i, nmsed_classes_i,
       valid_detections_i) = _generate_detections_per_image(
           boxes[i],
           scores[i],
           max_total_size,
           nms_iou_threshold,
           score_threshold,
           pre_nms_num_boxes)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
      valid_detections.append(valid_detections_i)
  nmsed_boxes = tf.stack(nmsed_boxes, axis=0)
  nmsed_scores = tf.stack(nmsed_scores, axis=0)
  nmsed_classes = tf.stack(nmsed_classes, axis=0)
  valid_detections = tf.stack(valid_detections, axis=0)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_per_image(boxes,
                                   scores,
                                   max_total_size=100,
                                   nms_iou_threshold=0.3,
                                   score_threshold=0.05,
                                   pre_nms_num_boxes=5000):
  """Generate the final detections per image given the model outputs.

  Args:
    boxes: a tensor with shape [N, num_classes, 4] or [N, 1, 4], which box
      predictions on all feature levels. The N is the number of total anchors on
      all levels.
    scores: a tensor with shape [N, num_classes], which stacks class probability
      on all feature levels. The N is the number of total anchors on all levels.
      The num_classes is the number of classes predicted by the model. Note that
      the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nmsed_boxes: `float` Tensor of shape [max_total_size, 4] representing top
      detected boxes in [y1, x1, y2, x2].
    nmsed_scores: `float` Tensor of shape [max_total_size] representing sorted
      confidence scores for detected boxes. The values are between [0, 1].
    nmsed_classes: `int` Tensor of shape [max_total_size] representing classes
      for detected boxes.
    valid_detections: `int` Tensor of shape [1] only the top `valid_detections`
      boxes are valid detections.
  """
  nmsed_boxes = []
  nmsed_scores = []
  nmsed_classes = []
  num_classes_for_box = boxes.get_shape().as_list()[1]
  num_classes = scores.get_shape().as_list()[1]
  for i in range(num_classes):
    boxes_i = boxes[:, min(num_classes_for_box - 1, i)]
    scores_i = scores[:, i]

    # Obtains pre_nms_num_boxes before running NMS.
    scores_i, indices = tf.nn.top_k(
        scores_i, k=tf.minimum(tf.shape(scores_i)[-1], pre_nms_num_boxes))
    boxes_i = tf.gather(boxes_i, indices)

    (nmsed_indices_i,
     nmsed_num_valid_i) = tf.image.non_max_suppression_padded(
         tf.cast(boxes_i, tf.float32),
         tf.cast(scores_i, tf.float32),
         max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_to_max_output_size=True,
         name='nms_detections_' + str(i))
    nmsed_boxes_i = tf.gather(boxes_i, nmsed_indices_i)
    nmsed_scores_i = tf.gather(scores_i, nmsed_indices_i)
    # Sets scores of invalid boxes to -1.
    nmsed_scores_i = tf.where(
        tf.less(tf.range(max_total_size), [nmsed_num_valid_i]),
        nmsed_scores_i, -tf.ones_like(nmsed_scores_i))
    nmsed_classes_i = tf.fill([max_total_size], i)
    nmsed_boxes.append(nmsed_boxes_i)
    nmsed_scores.append(nmsed_scores_i)
    nmsed_classes.append(nmsed_classes_i)

  # Concats results from all classes and sort them.
  nmsed_boxes = tf.concat(nmsed_boxes, axis=0)
  nmsed_scores = tf.concat(nmsed_scores, axis=0)
  nmsed_classes = tf.concat(nmsed_classes, axis=0)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_total_size, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices)
  nmsed_classes = tf.gather(nmsed_classes, indices)
  valid_detections = tf.reduce_sum(
      tf.cast(tf.greater(nmsed_scores, -1), tf.int32))
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _select_top_k_scores(scores_in, pre_nms_num_detections):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: a Tensor with shape [batch_size, N, num_classes], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  _, num_anchors, num_classes = scores_in.get_shape().as_list()
  scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
  scores_trans = tf.reshape(scores_trans, [-1, num_anchors])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores_trans, k=pre_nms_num_detections, sorted=True)

  top_k_scores = tf.reshape(
      top_k_scores, [-1, num_classes, pre_nms_num_detections])
  top_k_indices = tf.reshape(
      top_k_indices, [-1, num_classes, pre_nms_num_detections])

  return (tf.transpose(top_k_scores, [0, 2, 1]),
          tf.transpose(top_k_indices, [0, 2, 1]))


def _generate_detections_v2(boxes,
                            scores,
                            max_total_size=100,
                            nms_iou_threshold=0.3,
                            score_threshold=0.05,
                            pre_nms_num_boxes=5000):
  """Generate the final detections given the model outputs.

  This uses classes unrolling with while loop based NMS, could be parralled
  at batch dimension.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or [batch_size,
      N, 1, 4], which box predictions on all feature levels. The N is the number
      of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which stacks class
      probability on all feature levels. The N is the number of total anchors on
      all levels. The num_classes is the number of classes predicted by the
      model. Note that the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nmsed_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nmsed_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nmsed_classes: `int` Tensor of shape [batch_size, max_total_size]
      representing classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    num_classes_for_boxes = tf.shape(boxes)[2]
    total_anchors = tf.shape(scores)[1]
    num_classes = scores.get_shape().as_list()[2]
    # Selects top pre_nms_num scores and indices before NMS.
    scores, indices = _select_top_k_scores(
        scores, tf.minimum(total_anchors, pre_nms_num_boxes))
    for i in range(num_classes):
      boxes_i = boxes[:, :, tf.minimum(num_classes_for_boxes - 1, i), :]
      scores_i = scores[:, :, i]
      # Obtains pre_nms_num_boxes before running NMS.
      boxes_i = tf.gather(boxes_i, indices[:, :, i], batch_dims=1, axis=1)

      # Filter out scores.
      boxes_i, scores_i = box_utils.filter_boxes_by_scores(
          boxes_i, scores_i, min_score_threshold=score_threshold)

      (nmsed_scores_i, nmsed_boxes_i) = nms.sorted_non_max_suppression_padded(
          tf.cast(scores_i, tf.float32),
          tf.cast(boxes_i, tf.float32),
          max_total_size,
          iou_threshold=nms_iou_threshold)
      nmsed_classes_i = tf.fill(tf.shape(nmsed_scores_i), i)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
  nmsed_boxes = tf.concat(nmsed_boxes, axis=1)
  nmsed_scores = tf.concat(nmsed_scores, axis=1)
  nmsed_classes = tf.concat(nmsed_classes, axis=1)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_total_size, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices, batch_dims=1, axis=1)
  nmsed_classes = tf.gather(nmsed_classes, indices, batch_dims=1)
  valid_detections = tf.reduce_sum(
      input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32), axis=1)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_batched(boxes,
                                 scores,
                                 max_total_size,
                                 nms_iou_threshold,
                                 score_threshold):
  """Generates detected boxes with scores and classes for one-stage detector.

  The function takes output of multi-level ConvNets and anchor boxes and
  generates detected boxes. Note that this used batched nms, which is not
  supported on TPU currently.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or
      [batch_size, N, 1, 4], which box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.

  Returns:
    nmsed_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nmsed_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nmsed_classes: `int` Tensor of shape [batch_size, max_total_size]
      representing classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    # TODO(tsungyi): Removes normalization/denomalization once the
    # tf.image.combined_non_max_suppression is coordinate system agnostic.
    # Normalizes maximum box cooridinates to 1.
    normalizer = tf.reduce_max(boxes)
    boxes /= normalizer
    (nmsed_boxes, nmsed_scores, nmsed_classes,
     valid_detections) = tf.image.combined_non_max_suppression(
         tf.cast(boxes, tf.float32),
         tf.cast(scores, tf.float32),
         max_output_size_per_class=max_total_size,
         max_total_size=max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_per_class=False,
     )
    # De-normalizes box cooridinates.
    nmsed_boxes *= normalizer
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


class MultilevelDetectionGenerator(object):
  """Generates detected boxes with scores and classes for one-stage detector."""

  def __init__(self, min_level, max_level, params):
    self._min_level = min_level
    self._max_level = max_level
    self._apply_nms = params.apply_nms
    self._apply_sigmoid = params.apply_sigmoid
    self._generate_detections = generate_detections_factory(params)

  def __call__(self, box_outputs, class_outputs, anchor_boxes, image_shape):
    # Collects outputs from all levels into a list.
    boxes = []
    encoded_boxes = []
    scores = []
    for i in range(self._min_level, self._max_level + 1):
      _, feature_h, feature_w, num_predicted_corners = (
          box_outputs[i].get_shape().as_list())
      num_anchors_per_locations = num_predicted_corners // 4
      num_classes = (class_outputs[i].get_shape().as_list()[-1] //
                     num_anchors_per_locations)
      num_anchors = feature_h * feature_w  * num_anchors_per_locations

      scores_i = tf.reshape(class_outputs[i], [-1, num_anchors, num_classes])
      if self._apply_sigmoid:
        # Applies score transformation.
        scores_i = tf.sigmoid(scores_i)

      # Remove the implicit background class.
      scores_i = tf.slice(scores_i, [0, 0, 1], [-1, -1, -1])

      # Box decoding.
      # The anchor boxes are shared for all data in a batch.
      # One stage detector only supports class agnostic box regression.
      anchor_boxes_i = tf.reshape(anchor_boxes[i], [-1, num_anchors, 4])
      box_outputs_i = tf.reshape(box_outputs[i], [-1, num_anchors, 4])
      encoded_boxes.append(box_outputs_i)
      boxes_i = box_utils.decode_boxes(box_outputs_i, anchor_boxes_i)

      # Box clipping.
      boxes_i = box_utils.clip_boxes(boxes_i, image_shape)

      boxes.append(boxes_i)
      scores.append(scores_i)
    boxes = tf.concat(boxes, axis=1)
    boxes = tf.expand_dims(boxes, axis=2)
    encoded_boxes = tf.concat(encoded_boxes, axis=1)
    scores = tf.concat(scores, axis=1)

    if not self._apply_nms:
      return {
          'raw_boxes': boxes,
          'raw_encoded_boxes': encoded_boxes,
          'raw_scores': scores,
      }

    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        self._generate_detections(boxes, scores))

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return {
        'num_detections': valid_detections,
        'detection_boxes': nmsed_boxes,
        'detection_classes': nmsed_classes,
        'detection_scores': nmsed_scores,
    }


class GenericDetectionGenerator(object):
  """Generates the final detected boxes with scores and classes."""

  def __init__(self, params):
    self._apply_nms = params.apply_nms
    self._generate_detections = generate_detections_factory(params)

  def __call__(self, box_outputs, class_outputs, anchor_boxes, image_shape,
               regression_weights=None, bbox_per_class=True):
    """Generate final detections.

    Args:
      box_outputs: a tensor of shape of [batch_size, K, num_classes * 4]
        representing the class-specific box coordinates relative to anchors.
      class_outputs: a tensor of shape of [batch_size, K, num_classes]
        representing the class logits before applying score activiation.
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.
      regression_weights: A list of four float numbers to scale coordinates.
      bbox_per_class: A `bool`. If True, perform per-class box regression.

    Returns:
      nmsed_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      nmsed_scores: `float` Tensor of shape [batch_size, max_total_size]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      nmsed_classes: `int` Tensor of shape [batch_size, max_total_size]
        representing classes for detected boxes.
      valid_detections: `int` Tensor of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
    """
    class_outputs = tf.nn.softmax(class_outputs, axis=-1)

    # Removes the background class.
    class_outputs_shape = tf.shape(class_outputs)
    num_locations = class_outputs_shape[1]
    num_classes = class_outputs_shape[-1]

    class_outputs = tf.slice(class_outputs, [0, 0, 1], [-1, -1, -1])
    if bbox_per_class:
      num_detections = num_locations * (num_classes - 1)
      box_outputs = tf.reshape(box_outputs, [-1, num_locations, num_classes, 4])
      box_outputs = tf.slice(
          box_outputs, [0, 0, 1, 0], [-1, -1, -1, -1])
      anchor_boxes = tf.tile(
          tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])
      box_outputs = tf.reshape(box_outputs, [-1, num_detections, 4])
      anchor_boxes = tf.reshape(anchor_boxes, [-1, num_detections, 4])

    # Box decoding.
    if regression_weights is None:
      regression_weights = [10.0, 10.0, 5.0, 5.0]
    decoded_boxes = box_utils.decode_boxes(
        box_outputs, anchor_boxes, weights=regression_weights)

    # Box clipping
    decoded_boxes = box_utils.clip_boxes(decoded_boxes, image_shape)

    if bbox_per_class:
      decoded_boxes = tf.reshape(decoded_boxes,
                                 [-1, num_locations, num_classes - 1, 4])
    else:
      decoded_boxes = tf.expand_dims(decoded_boxes, axis=2)

    if not self._apply_nms:
      return {
          'raw_boxes': decoded_boxes,
          'raw_scores': class_outputs,
      }

    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        self._generate_detections(decoded_boxes, class_outputs))

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return {
        'num_detections': valid_detections,
        'detection_boxes': nmsed_boxes,
        'detection_classes': nmsed_classes,
        'detection_scores': nmsed_scores,
    }
