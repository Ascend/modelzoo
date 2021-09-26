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
"""Tensorflow implementation of non max suppression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow.compat.v1 as tf

from utils import box_utils


NMS_TILE_SIZE = 512


def _self_suppression(iou, _, iou_sum, iou_threshold):
  """Suppress boxes in the same tile.

     Compute boxes that cannot be suppressed by others (i.e.,
     can_suppress_others), and then use them to suppress boxes in the same tile.

  Args:
    iou: a tensor of shape [batch_size, num_boxes_with_padding].
    iou_sum: a scalar tensor.
    iou_threshold: a scalar tensor.

  Returns:
    iou_suppressed: a tensor of shape [batch_size, num_boxes_with_padding].
    iou_diff: a scalar tensor representing whether any box is supressed in
      this step.
    iou_sum_new: a scalar tensor of shape [batch_size] that represents
      the iou sum after suppression.
    iou_threshold: a scalar tensor.
  """
  batch_size = tf.shape(iou)[0]
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(iou, 1) <= iou_threshold, [batch_size, -1, 1]),
      iou.dtype)
  iou_after_suppression = tf.reshape(
      tf.cast(
          tf.reduce_max(can_suppress_others * iou, 1) <= iou_threshold,
          iou.dtype), [batch_size, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(iou_after_suppression, [1, 2])
  return [
      iou_after_suppression,
      tf.reduce_any(iou_sum - iou_sum_new > iou_threshold), iou_sum_new,
      iou_threshold
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  """Suppress boxes between different tiles.

  Args:
    boxes: a tensor of shape [batch_size, num_boxes_with_padding, 4]
    box_slice: a tensor of shape [batch_size, _NMS_TILE_SIZE, 4]
    iou_threshold: a scalar tensor
    inner_idx: a scalar tensor representing the tile index of the tile that is
      used to supress box_slice

  Returns:
    boxes: unchanged boxes as input
    box_slice_after_suppression: box_slice after suppression
    iou_threshold: unchanged
  """
  batch_size = tf.shape(boxes)[0]
  new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  iou = box_utils.bbox_overlap(new_slice, box_slice)
  box_slice_after_suppression = tf.expand_dims(
      tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
      2) * box_slice
  return boxes, box_slice_after_suppression, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  num_tiles = tf.shape(boxes)[1] // NMS_TILE_SIZE
  batch_size = tf.shape(boxes)[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      _cross_suppression, [boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = box_utils.bbox_overlap(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _, _ = tf.while_loop(
      lambda _iou, loop_condition, _iou_sum, _: loop_condition,
      _self_suppression,
      [iou, tf.constant(True),
       tf.reduce_sum(iou, [1, 2]), iou_threshold])
  suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(tf.expand_dims(
      box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
          boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = tf.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def sorted_non_max_suppression_padded(scores,
                                      boxes,
                                      max_output_size,
                                      iou_threshold):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  While a serial NMS algorithm iteratively uses the highest-scored unprocessed
  box to suppress boxes, this algorithm uses many boxes to suppress other boxes
  in parallel. The key idea is to partition boxes into tiles based on their
  score and suppresses boxes tile by tile, thus achieving parallelism within a
  tile. The tile size determines the degree of parallelism.

  In cross suppression (using boxes of tile A to suppress boxes of tile B),
  all boxes in A can independently suppress boxes in B.

  Self suppression (suppressing boxes of the same tile) needs to be iteratively
  applied until there's no more suppression. In each iteration, boxes that
  cannot be suppressed are used to suppress boxes in the same tile.

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      # in parallel suppress boxes in box_tile using boxes from suppressing_tile
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.

  Returns:
    nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
      dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
      same dtype as input boxes.
  """
  batch_size = tf.shape(boxes)[0]
  num_boxes = tf.shape(boxes)[1]
  pad = tf.cast(
      tf.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
      tf.int32) * NMS_TILE_SIZE - num_boxes
  boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
  scores = tf.pad(
      tf.cast(scores, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
  num_boxes += pad

  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(
        tf.reduce_min(output_size) < max_output_size,
        idx < num_boxes // NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = tf.while_loop(
      _loop_cond, _suppression_loop_body, [
          boxes, iou_threshold,
          tf.zeros([batch_size], tf.int32),
          tf.constant(0)
      ])
  idx = num_boxes - tf.cast(
      tf.nn.top_k(
          tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
          tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
      tf.int32)
  idx = tf.minimum(idx, num_boxes - 1)
  idx = tf.reshape(
      idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
  boxes = tf.reshape(
      tf.gather(tf.reshape(boxes, [-1, 4]), idx),
      [batch_size, max_output_size, 4])
  boxes_mask = tf.less(
      tf.reshape(tf.range(max_output_size), [1, -1, 1]),
      tf.reshape(output_size, [-1, 1, 1]))
  boxes = boxes * tf.cast(boxes_mask, boxes.dtype)
  scores = tf.reshape(
      tf.gather(tf.reshape(scores, [-1, 1]), idx),
      [batch_size, max_output_size])
  scores_mask = tf.less(
      tf.reshape(tf.range(max_output_size), [1, -1]),
      tf.reshape(output_size, [-1, 1]))
  scores = tf.where(scores_mask, scores,
                    tf.ones_like(scores, scores.dtype) * -1)
  return scores, boxes
