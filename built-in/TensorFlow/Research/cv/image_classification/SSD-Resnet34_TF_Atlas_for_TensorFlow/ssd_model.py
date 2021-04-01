# Copyright 2018 Google. All Rights Reserved.
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
"""Model defination for the SSD Model.

Defines model_fn of SSD for TF Estimator. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

from object_detection import box_coder
from object_detection import box_list
from object_detection import faster_rcnn_box_coder

from tensorflow.python.estimator import model_fn as model_fn_lib

import dataloader
import ssd_architecture
import ssd_constants


def get_rank_size():
    return int(os.environ['RANK_SIZE'])

def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
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
  scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores_trans, k=pre_nms_num_detections, sorted=True)

  return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
      top_k_indices, [0, 2, 1])


def concat_outputs(cls_outputs, box_outputs):
  """Concatenate predictions into a single tensor.

  This function takes the dicts of class and box prediction tensors and
  concatenates them into a single tensor for comparison with the ground truth
  boxes and class labels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width,
      num_anchors * num_classses].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
  Returns:
    concatenanted cls_outputs and box_outputs.
  """
  assert set(cls_outputs.keys()) == set(box_outputs.keys())

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())
  batch_size = int(cls_outputs[keys[0]].shape[0])

  flat_cls = []
  flat_box = []

  for i, k in enumerate(keys):
    # TODO(taylorrobie): confirm that this reshape, transpose,
    # reshape is correct.
    scale = ssd_constants.FEATURE_SIZES[i] # 不同特征尺度, 38,19,10,5,3,1
    split_shape = (ssd_constants.NUM_DEFAULTS[i], ssd_constants.NUM_CLASSES) # （4,81）、（6,81）...
    assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape # (32,38,38)+ (4,81)=(32,38,38,4,81)
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1]) # (32, 38^2 * 4, 81)
    flat_cls.append(tf.reshape(tf.reshape(
        cls_outputs[k], intermediate_shape), final_shape))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4) # (4,4), (6,4)...
    assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape # (32, 19,19) + (6,4) 为避免歧义，以第二个default box为例
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1]) # (32, 19^2 * 6, 4)
    flat_box.append(tf.reshape(tf.reshape(
        box_outputs[k], intermediate_shape), final_shape))

  return tf.concat(flat_cls, axis=1), tf.concat(flat_box, axis=1)


def _localization_loss(pred_locs, gt_locs, gt_labels, num_matched_boxes):
  """Computes the localization loss.

  Computes the localization loss using smooth l1 loss.
  Args:
    pred_locs: a dict from index to tensor of predicted locations. The shape
      of each tensor is [batch_size, num_anchors, 4].
    gt_locs: a list of tensors representing box regression targets in
      [batch_size, num_anchors, 4].
    gt_labels: a list of tensors that represents the classification groundtruth
      targets. The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets, used as the loss normalizater. The shape is [batch_size].
  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  keys = sorted(pred_locs.keys())
  box_loss = 0
  for i, k in enumerate(keys):
    gt_label = gt_labels[i]
    gt_loc = gt_locs[i]
    pred_loc = tf.reshape(pred_locs[k], gt_loc.shape)
    mask = tf.greater(gt_label, 0)
    float_mask = tf.cast(mask, tf.float32)

    smooth_l1 = tf.reduce_sum(
        tf.losses.huber_loss(
            gt_loc, pred_loc, reduction=tf.losses.Reduction.NONE),
        axis=-1)
    smooth_l1 = tf.multiply(smooth_l1, float_mask)
    box_loss = box_loss + tf.reduce_sum(
        smooth_l1, axis=list(range(1, smooth_l1.shape.ndims)))

  # TODO(taylorrobie): Confirm that normalizing by the number of boxes matches
  # reference
  return tf.reduce_mean(box_loss / num_matched_boxes)


@tf.custom_gradient
def _softmax_cross_entropy(logits, label):
  """Helper function to compute softmax cross entropy loss."""
  shifted_logits = logits - tf.expand_dims(tf.reduce_max(logits, -1), -1)
  exp_shifted_logits = tf.math.exp(shifted_logits)
  sum_exp = tf.reduce_sum(exp_shifted_logits, -1)
  log_sum_exp = tf.math.log(sum_exp)
  one_hot_label = tf.one_hot(label, ssd_constants.NUM_CLASSES)
  shifted_logits = tf.reduce_sum(shifted_logits * one_hot_label, -1)
  loss = log_sum_exp - shifted_logits

  def grad(dy):
    return (exp_shifted_logits / tf.expand_dims(sum_exp, -1) -
            one_hot_label) * tf.expand_dims(dy, -1), dy

  return loss, grad


def _classification_loss(pred_labels, gt_labels, num_matched_boxes):
  """Computes the classification loss.

  Computes the classification loss with hard negative mining.
  Args:
    pred_labels: a dict from index to tensor of predicted class. The shape
      of the tensor is [batch_size, num_anchors, num_classes].
    gt_labels: a list of tensor that represents the classification groundtruth
      targets. The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets. This is used as the loss normalizater.
  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  keys = sorted(pred_labels.keys())
  batch_size = gt_labels[0].shape[0]
  cross_entropy = []
  for i, k in enumerate(keys):
    gt_label = gt_labels[i]
    pred_label = tf.reshape(
        pred_labels[k],
        gt_label.get_shape().as_list() + [ssd_constants.NUM_CLASSES])
    cross_entropy.append(
        tf.reshape(
            _softmax_cross_entropy(pred_label, gt_label), [batch_size, -1]))


  # Put the rest of the loss computation on one device to avoid excessive
  # communication inside topk_mask with spatial partition
  #with tf.device(tf.contrib.tpu.core(0)):
  cross_entropy = tf.concat(cross_entropy, 1)
  gt_label = tf.concat([tf.reshape(l, [batch_size, -1]) for l in gt_labels],
                         1)
  mask = tf.greater(gt_label, 0)
  float_mask = tf.cast(mask, tf.float32)

    # Hard example mining
  neg_masked_cross_entropy = cross_entropy * (1 - float_mask)


  value1, _ = tf.math.top_k(neg_masked_cross_entropy, k=4096)
  kth1 = tf.reduce_min(value1, 1, keepdims=True)
  mask1 = tf.cast(tf.less(neg_masked_cross_entropy, kth1), tf.float32)

  value2, _ = tf.math.top_k(tf.multiply(neg_masked_cross_entropy, mask1), k=4096)
  kth2 = tf.reduce_min(value2, 1, keepdims=True)
  mask2 = tf.cast(tf.less(neg_masked_cross_entropy, kth2), tf.float32)

  value3, _ = tf.math.top_k(tf.multiply(neg_masked_cross_entropy, mask2), k=540)

  value = tf.concat([value1, value2, value3], axis=1)

  num_neg_boxes = tf.minimum(
          tf.to_int32(num_matched_boxes) * ssd_constants.NEGS_PER_POSITIVE, 8731)
  large_neg_ce = tf.batch_gather(value, num_neg_boxes[:, tf.newaxis])
  top_k_neg_mask = tf.cast(tf.greater_equal(neg_masked_cross_entropy, large_neg_ce), tf.float32)



  class_loss = tf.reduce_sum(
        tf.multiply(cross_entropy, float_mask + top_k_neg_mask), axis=1)


    # TODO(taylorrobie): Confirm that normalizing by the number of boxes matches
    # reference
  return tf.reduce_mean(class_loss / num_matched_boxes)


def detection_loss(cls_outputs, box_outputs, labels):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
  Returns:
    total_loss: a float32 representing total loss reducing from class and box
      losses from all levels.
    cls_loss: a float32 representing total class loss.
    box_loss: a float32 representing total box regression loss.
  """
  if isinstance(labels[ssd_constants.BOXES], dict):
    gt_boxes = list(labels[ssd_constants.BOXES].values())
    gt_classes = list(labels[ssd_constants.CLASSES].values())
  else:
    gt_boxes = [labels[ssd_constants.BOXES]]
    gt_classes = [labels[ssd_constants.CLASSES]]
    cls_outputs, box_outputs = concat_outputs(cls_outputs, box_outputs)
    cls_outputs = {'flatten': cls_outputs}
    box_outputs = {'flatten': box_outputs}

  box_loss = _localization_loss(box_outputs, gt_boxes, gt_classes,
                                labels[ssd_constants.NUM_MATCHED_BOXES])
  class_loss = _classification_loss(cls_outputs, gt_classes,
                                    labels[ssd_constants.NUM_MATCHED_BOXES])

  return class_loss + box_loss, class_loss, box_loss


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  Args:
    params: a parameter dictionary that includes learning_rate, lr_warmup_epoch,
      first_lr_drop_epoch, and second_lr_drop_epoch.
  """
  batch_size = params['batch_size']
  # Learning rate is proportional to the batch size
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size // get_rank_size()
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['cos_decay_step'] = int(
      params['cos_decay_epoch'] * steps_per_epoch)


def learning_rate_schedule(params, global_step):
  """Handles learning rate scaling, linear warmup, and learning rate decay.

  Args:
    params: A dictionary that defines hyperparameters of model.
    global_step: A tensor representing current global step.

  Returns:
    A tensor representing current learning rate.
  """
  base_learning_rate = params['base_learning_rate']
  lr_warmup_step = params['lr_warmup_step']
  cos_decay_step = params['cos_decay_step']
  batch_size = params['batch_size']
  scaling_factor = get_rank_size() * batch_size / ssd_constants.DEFAULT_BATCH_SIZE
  adjusted_learning_rate = base_learning_rate * scaling_factor
  learning_rate = (tf.cast(global_step, dtype=tf.float32) /
                   lr_warmup_step) * adjusted_learning_rate

  learning_rate = tf.where(global_step < lr_warmup_step, learning_rate,
                           tf.train.cosine_decay(adjusted_learning_rate, global_step, cos_decay_step, alpha=0.01))

  return learning_rate


class ExamplesPerSecondHook(tf.train.SessionRunHook):
  def __init__(
      self,
      batch_size,
      lr=0,
      loss=0,
      every_n_steps=100,
      every_n_secs=None,):


    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')

    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size
    self._lr = lr
    self._loss = loss

  def begin(self):
    self._global_step_tensor = tf.compat.v1.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs([self._global_step_tensor, self._lr, self._loss])

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step, lr, loss = run_values.results
    if self._timer.should_trigger_for_step(global_step):

      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        current_examples_per_sec = steps_per_sec * self._batch_size
        tf.logging.info('%s: %g, %s: %s, %s: %s', 'FPS', current_examples_per_sec, 'learning rate', lr, 'loss', loss)



def _model_fn(features, labels, mode, params, model):
  """Model defination for the SSD model based on ResNet-50.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the SSD model outputs class logits and box regression outputs.

  Returns:
    spec: the EstimatorSpec or TPUEstimatorSpec to run training, evaluation,
      or prediction.
  """
  if mode == tf.estimator.ModeKeys.PREDICT:
    labels = features
    features = labels.pop('image')

  features -= tf.constant(
        ssd_constants.NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=features.dtype)

  features /= tf.constant(
        ssd_constants.NORMALIZATION_STD, shape=[1, 1, 3], dtype=features.dtype)

  def _model_outputs():
    return model(
        features, params, is_training_bn=(mode == tf.estimator.ModeKeys.TRAIN))


  cls_outputs, box_outputs = _model_outputs()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    flattened_cls, flattened_box = concat_outputs(cls_outputs, box_outputs)
    ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=ssd_constants.BOX_CODER_SCALES)

    anchors = box_list.BoxList(
        tf.convert_to_tensor(dataloader.DefaultBoxes()('ltrb')))

    decoded_boxes = box_coder.batch_decode(
        encoded_boxes=flattened_box, box_coder=ssd_box_coder, anchors=anchors)

    pred_scores = tf.nn.softmax(flattened_cls, axis=2)

    pred_scores, indices = select_top_k_scores(pred_scores,
                                               ssd_constants.MAX_NUM_EVAL_BOXES)

    predictions = dict(
          labels,
          indices=indices,
          pred_scores=pred_scores,
          pred_box=decoded_boxes,
    )

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Load pretrained model from checkpoint.
  if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
          '/': 'resnet%s/' % ssd_constants.RESNET_DEPTH,
      })
      return tf.train.Scaffold()
  else:
    scaffold_fn = None

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)
  # cls_loss and box_loss are for logging. only total_loss is optimized.
  total_loss, cls_loss, box_loss = detection_loss(
      cls_outputs, box_outputs, labels)

  total_loss += params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    total_loss_t = tf.reduce_mean(tf.reshape(total_loss, [1]))
    cls_loss_t = tf.reduce_mean(tf.reshape(cls_loss, [1]))
    box_loss_t = tf.reduce_mean(tf.reshape(box_loss, [1]))
    learning_rate_t = tf.reduce_mean(tf.reshape(learning_rate, [1]))
    tf.summary.scalar('total_loss', total_loss_t)
    tf.summary.scalar('cls_loss_t', cls_loss_t)
    tf.summary.scalar('box_loss_t', box_loss_t)
    tf.summary.scalar('learning_rate_t', learning_rate_t)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=ssd_constants.MOMENTUM)
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    optimizer = NPUDistributedOptimizer(optimizer)  # 使用NPU分布式计算，更新梯度

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    examples_sec_hook = ExamplesPerSecondHook(get_rank_size() * 32, learning_rate, total_loss)

    train_op = tf.group(optimizer.minimize(total_loss, global_step),
                        update_ops)
    return model_fn_lib.EstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold_fn(),
        training_hooks=[examples_sec_hook])

  if mode == tf.estimator.ModeKeys.EVAL:
    raise NotImplementedError


def ssd_model_fn(features, labels, mode, params):
  """SSD model."""
  return _model_fn(features, labels, mode, params, model=ssd_architecture.ssd)


def default_hparams():
  # TODO(taylorrobie): replace params useages with global constants.
  return tf.contrib.training.HParams(

      num_examples_per_epoch=120000,
      lr_warmup_epoch=0.8,
      cos_decay_epoch=106,
      weight_decay=ssd_constants.WEIGHT_DECAY,
      base_learning_rate=ssd_constants.BASE_LEARNING_RATE,
      eval_every_checkpoint=False,
      transpose_input=False,
      use_cocoeval_cc=False
  )
