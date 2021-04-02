
'Classification and regression loss functions for object detection.\n\nLocalization losses:\n * WeightedL2LocalizationLoss\n * WeightedSmoothL1LocalizationLoss\n * WeightedIOULocalizationLoss\n\nClassification losses:\n * WeightedSigmoidClassificationLoss\n * WeightedSoftmaxClassificationLoss\n * WeightedSoftmaxClassificationAgainstLogitsLoss\n * BootstrappedSigmoidClassificationLoss\n'
from npu_bridge.npu_init import *
from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import ops
slim = tf.contrib.slim

class Loss(object):
    'Abstract base class for loss functions.'
    __metaclass__ = ABCMeta

    def __call__(self, prediction_tensor, target_tensor, ignore_nan_targets=False, losses_mask=None, scope=None, **params):
        "Call the loss function.\n\n    Args:\n      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]\n        representing predicted quantities.\n      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing\n        regression or classification targets.\n      ignore_nan_targets: whether to ignore nan targets in the loss computation.\n        E.g. can be used if the target tensor is missing groundtruth data that\n        shouldn't be factored into the loss.\n      losses_mask: A [batch] boolean tensor that indicates whether losses should\n        be applied to individual images in the batch. For elements that\n        are True, corresponding prediction, target, and weight tensors will be\n        removed prior to loss computation. If None, no filtering will take place\n        prior to loss computation.\n      scope: Op scope name. Defaults to 'Loss' if None.\n      **params: Additional keyword arguments for specific implementations of\n              the Loss.\n\n    Returns:\n      loss: a tensor representing the value of the loss function.\n    "
        with tf.name_scope(scope, 'Loss', [prediction_tensor, target_tensor, params]) as scope:
            if ignore_nan_targets:
                target_tensor = tf.where(tf.is_nan(target_tensor), prediction_tensor, target_tensor)
            if (losses_mask is not None):
                tensor_multiplier = self._get_loss_multiplier_for_tensor(prediction_tensor, losses_mask)
                prediction_tensor *= tensor_multiplier
                target_tensor *= tensor_multiplier
                if ('weights' in params):
                    params['weights'] = tf.convert_to_tensor(params['weights'])
                    weights_multiplier = self._get_loss_multiplier_for_tensor(params['weights'], losses_mask)
                    params['weights'] *= weights_multiplier
            return self._compute_loss(prediction_tensor, target_tensor, **params)

    def _get_loss_multiplier_for_tensor(self, tensor, losses_mask):
        loss_multiplier_shape = tf.stack(([(- 1)] + ([1] * (len(tensor.shape) - 1))))
        return tf.cast(tf.reshape(losses_mask, loss_multiplier_shape), tf.float32)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        'Method to be overridden by implementations.\n\n    Args:\n      prediction_tensor: a tensor representing predicted quantities\n      target_tensor: a tensor representing regression or classification targets\n      **params: Additional keyword arguments for specific implementations of\n              the Loss.\n\n    Returns:\n      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per\n        anchor\n    '
        pass

class WeightedL2LocalizationLoss(Loss):
    'L2 localization loss function with anchorwise output support.\n\n  Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2\n  '

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        code_size] representing the (encoded) predicted locations of objects.\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        code_size] representing the regression targets\n      weights: a float tensor of shape [batch_size, num_anchors]\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors] tensor\n        representing the value of the loss function.\n    '
        weighted_diff = ((prediction_tensor - target_tensor) * tf.expand_dims(weights, 2))
        square_diff = (0.5 * tf.square(weighted_diff))
        return tf.reduce_sum(square_diff, 2)

class WeightedSmoothL1LocalizationLoss(Loss):
    'Smooth L1 localization loss function aka Huber Loss..\n\n  The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and\n  delta * (|x|- 0.5*delta) otherwise, where x is the difference between\n  predictions and target.\n\n  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)\n  '

    def __init__(self, delta=1.0):
        'Constructor.\n\n    Args:\n      delta: delta for smooth L1 loss.\n    '
        self._delta = delta

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        code_size] representing the (encoded) predicted locations of objects.\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        code_size] representing the regression targets\n      weights: a float tensor of shape [batch_size, num_anchors]\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors] tensor\n        representing the value of the loss function.\n    '
        return tf.reduce_sum(tf.losses.huber_loss(target_tensor, prediction_tensor, delta=self._delta, weights=tf.expand_dims(weights, axis=2), loss_collection=None, reduction=tf.losses.Reduction.NONE), axis=2)

class WeightedIOULocalizationLoss(Loss):
    'IOU localization loss function.\n\n  Sums the IOU for corresponding pairs of predicted/groundtruth boxes\n  and for each pair assign a loss of 1 - IOU.  We then compute a weighted\n  sum over all pairs which is returned as the total loss.\n  '

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors, 4]\n        representing the decoded predicted boxes\n      target_tensor: A float tensor of shape [batch_size, num_anchors, 4]\n        representing the decoded target boxes\n      weights: a float tensor of shape [batch_size, num_anchors]\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors] tensor\n        representing the value of the loss function.\n    '
        predicted_boxes = box_list.BoxList(tf.reshape(prediction_tensor, [(- 1), 4]))
        target_boxes = box_list.BoxList(tf.reshape(target_tensor, [(- 1), 4]))
        per_anchor_iou_loss = (1.0 - box_list_ops.matched_iou(predicted_boxes, target_boxes))
        return (tf.reshape(weights, [(- 1)]) * per_anchor_iou_loss)

class WeightedSigmoidClassificationLoss(Loss):
    'Sigmoid cross entropy classification loss function.'

    def _compute_loss(self, prediction_tensor, target_tensor, weights, class_indices=None):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing the predicted logits for each class\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing one-hot encoded classification targets\n      weights: a float tensor of shape, either [batch_size, num_anchors,\n        num_classes] or [batch_size, num_anchors, 1]. If the shape is\n        [batch_size, num_anchors, 1], all the classses are equally weighted.\n      class_indices: (Optional) A 1-D integer tensor of class indices.\n        If provided, computes loss only for the specified class indices.\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors, num_classes]\n        representing the value of the loss function.\n    '
        if (class_indices is not None):
            weights *= tf.reshape(ops.indices_to_dense_vector(class_indices, tf.shape(prediction_tensor)[2]), [1, 1, (- 1)])
        per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
        return (per_entry_cross_ent * weights)

class SigmoidFocalClassificationLoss(Loss):
    'Sigmoid focal cross entropy loss.\n\n  Focal loss down-weights well classified examples and focusses on the hard\n  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.\n  '

    def __init__(self, gamma=2.0, alpha=0.25):
        'Constructor.\n\n    Args:\n      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.\n      alpha: optional alpha weighting factor to balance positives vs negatives.\n    '
        self._alpha = alpha
        self._gamma = gamma

    def _compute_loss(self, prediction_tensor, target_tensor, weights, class_indices=None):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing the predicted logits for each class\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing one-hot encoded classification targets\n      weights: a float tensor of shape, either [batch_size, num_anchors,\n        num_classes] or [batch_size, num_anchors, 1]. If the shape is\n        [batch_size, num_anchors, 1], all the classses are equally weighted.\n      class_indices: (Optional) A 1-D integer tensor of class indices.\n        If provided, computes loss only for the specified class indices.\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors, num_classes]\n        representing the value of the loss function.\n    '
        if (class_indices is not None):
            weights *= tf.reshape(ops.indices_to_dense_vector(class_indices, tf.shape(prediction_tensor)[2]), [1, 1, (- 1)])
        per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
        prediction_probabilities = tf.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) + ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = tf.pow((1.0 - p_t), self._gamma)
        alpha_weight_factor = 1.0
        if (self._alpha is not None):
            alpha_weight_factor = ((target_tensor * self._alpha) + ((1 - target_tensor) * (1 - self._alpha)))
        focal_cross_entropy_loss = ((modulating_factor * alpha_weight_factor) * per_entry_cross_ent)
        return (focal_cross_entropy_loss * weights)

class WeightedSoftmaxClassificationLoss(Loss):
    'Softmax loss function.'

    def __init__(self, logit_scale=1.0):
        'Constructor.\n\n    Args:\n      logit_scale: When this value is high, the prediction is "diffused" and\n                   when this value is low, the prediction is made peakier.\n                   (default 1.0)\n\n    '
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing the predicted logits for each class\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing one-hot encoded classification targets\n      weights: a float tensor of shape, either [batch_size, num_anchors,\n        num_classes] or [batch_size, num_anchors, 1]. If the shape is\n        [batch_size, num_anchors, 1], all the classses are equally weighted.\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors]\n        representing the value of the loss function.\n    '
        weights = tf.reduce_mean(weights, axis=2)
        num_classes = prediction_tensor.get_shape().as_list()[(- 1)]
        prediction_tensor = tf.divide(prediction_tensor, self._logit_scale, name='scale_logit')
        per_row_cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(target_tensor, [(- 1), num_classes]), logits=tf.reshape(prediction_tensor, [(- 1), num_classes]))
        return (tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights)

class WeightedSoftmaxClassificationAgainstLogitsLoss(Loss):
    'Softmax loss function against logits.\n\n   Targets are expected to be provided in logits space instead of "one hot" or\n   "probability distribution" space.\n  '

    def __init__(self, logit_scale=1.0):
        'Constructor.\n\n    Args:\n      logit_scale: When this value is high, the target is "diffused" and\n                   when this value is low, the target is made peakier.\n                   (default 1.0)\n\n    '
        self._logit_scale = logit_scale

    def _scale_and_softmax_logits(self, logits):
        'Scale logits then apply softmax.'
        scaled_logits = tf.divide(logits, self._logit_scale, name='scale_logits')
        return tf.nn.softmax(scaled_logits, name='convert_scores')

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing the predicted logits for each class\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing logit classification targets\n      weights: a float tensor of shape, either [batch_size, num_anchors,\n        num_classes] or [batch_size, num_anchors, 1]. If the shape is\n        [batch_size, num_anchors, 1], all the classses are equally weighted.\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors]\n        representing the value of the loss function.\n    '
        weights = tf.reduce_mean(weights, axis=2)
        num_classes = prediction_tensor.get_shape().as_list()[(- 1)]
        target_tensor = self._scale_and_softmax_logits(target_tensor)
        prediction_tensor = tf.divide(prediction_tensor, self._logit_scale, name='scale_logits')
        per_row_cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(target_tensor, [(- 1), num_classes]), logits=tf.reshape(prediction_tensor, [(- 1), num_classes]))
        return (tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights)

class BootstrappedSigmoidClassificationLoss(Loss):
    'Bootstrapped sigmoid cross entropy classification loss function.\n\n  This loss uses a convex combination of training labels and the current model\'s\n  predictions as training targets in the classification loss. The idea is that\n  as the model improves over time, its predictions can be trusted more and we\n  can use these predictions to mitigate the damage of noisy/incorrect labels,\n  because incorrect labels are likely to be eventually highly inconsistent with\n  other stimuli predicted to have the same label by the model.\n\n  In "soft" bootstrapping, we use all predicted class probabilities, whereas in\n  "hard" bootstrapping, we use the single class favored by the model.\n\n  See also Training Deep Neural Networks On Noisy Labels with Bootstrapping by\n  Reed et al. (ICLR 2015).\n  '

    def __init__(self, alpha, bootstrap_type='soft'):
        "Constructor.\n\n    Args:\n      alpha: a float32 scalar tensor between 0 and 1 representing interpolation\n        weight\n      bootstrap_type: set to either 'hard' or 'soft' (default)\n\n    Raises:\n      ValueError: if bootstrap_type is not either 'hard' or 'soft'\n    "
        if ((bootstrap_type != 'hard') and (bootstrap_type != 'soft')):
            raise ValueError("Unrecognized bootstrap_type: must be one of 'hard' or 'soft.'")
        self._alpha = alpha
        self._bootstrap_type = bootstrap_type

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        'Compute loss function.\n\n    Args:\n      prediction_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing the predicted logits for each class\n      target_tensor: A float tensor of shape [batch_size, num_anchors,\n        num_classes] representing one-hot encoded classification targets\n      weights: a float tensor of shape, either [batch_size, num_anchors,\n        num_classes] or [batch_size, num_anchors, 1]. If the shape is\n        [batch_size, num_anchors, 1], all the classses are equally weighted.\n\n    Returns:\n      loss: a float tensor of shape [batch_size, num_anchors, num_classes]\n        representing the value of the loss function.\n    '
        if (self._bootstrap_type == 'soft'):
            bootstrap_target_tensor = ((self._alpha * target_tensor) + ((1.0 - self._alpha) * tf.sigmoid(prediction_tensor)))
        else:
            bootstrap_target_tensor = ((self._alpha * target_tensor) + ((1.0 - self._alpha) * tf.cast((tf.sigmoid(prediction_tensor) > 0.5), tf.float32)))
        per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=bootstrap_target_tensor, logits=prediction_tensor)
        return (per_entry_cross_ent * weights)

class HardExampleMiner(object):
    'Hard example mining for regions in a list of images.\n\n  Implements hard example mining to select a subset of regions to be\n  back-propagated. For each image, selects the regions with highest losses,\n  subject to the condition that a newly selected region cannot have\n  an IOU > iou_threshold with any of the previously selected regions.\n  This can be achieved by re-using a greedy non-maximum suppression algorithm.\n  A constraint on the number of negatives mined per positive region can also be\n  enforced.\n\n  Reference papers: "Training Region-based Object Detectors with Online\n  Hard Example Mining" (CVPR 2016) by Srivastava et al., and\n  "SSD: Single Shot MultiBox Detector" (ECCV 2016) by Liu et al.\n  '

    def __init__(self, num_hard_examples=64, iou_threshold=0.7, loss_type='both', cls_loss_weight=0.05, loc_loss_weight=0.06, max_negatives_per_positive=None, min_negatives_per_image=0):
        "Constructor.\n\n    The hard example mining implemented by this class can replicate the behavior\n    in the two aforementioned papers (Srivastava et al., and Liu et al).\n    To replicate the A2 paper (Srivastava et al), num_hard_examples is set\n    to a fixed parameter (64 by default) and iou_threshold is set to .7 for\n    running non-max-suppression the predicted boxes prior to hard mining.\n    In order to replicate the SSD paper (Liu et al), num_hard_examples should\n    be set to None, max_negatives_per_positive should be 3 and iou_threshold\n    should be 1.0 (in order to effectively turn off NMS).\n\n    Args:\n      num_hard_examples: maximum number of hard examples to be\n        selected per image (prior to enforcing max negative to positive ratio\n        constraint).  If set to None, all examples obtained after NMS are\n        considered.\n      iou_threshold: minimum intersection over union for an example\n        to be discarded during NMS.\n      loss_type: use only classification losses ('cls', default),\n        localization losses ('loc') or both losses ('both').\n        In the last case, cls_loss_weight and loc_loss_weight are used to\n        compute weighted sum of the two losses.\n      cls_loss_weight: weight for classification loss.\n      loc_loss_weight: weight for location loss.\n      max_negatives_per_positive: maximum number of negatives to retain for\n        each positive anchor. By default, num_negatives_per_positive is None,\n        which means that we do not enforce a prespecified negative:positive\n        ratio.  Note also that num_negatives_per_positives can be a float\n        (and will be converted to be a float even if it is passed in otherwise).\n      min_negatives_per_image: minimum number of negative anchors to sample for\n        a given image. Setting this to a positive number allows sampling\n        negatives in an image without any positive anchors and thus not biased\n        towards at least one detection per image.\n    "
        self._num_hard_examples = num_hard_examples
        self._iou_threshold = iou_threshold
        self._loss_type = loss_type
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._max_negatives_per_positive = max_negatives_per_positive
        self._min_negatives_per_image = min_negatives_per_image
        if (self._max_negatives_per_positive is not None):
            self._max_negatives_per_positive = float(self._max_negatives_per_positive)
        self._num_positives_list = None
        self._num_negatives_list = None

    def __call__(self, location_losses, cls_losses, decoded_boxlist_list, match_list=None):
        'Computes localization and classification losses after hard mining.\n\n    Args:\n      location_losses: a float tensor of shape [num_images, num_anchors]\n        representing anchorwise localization losses.\n      cls_losses: a float tensor of shape [num_images, num_anchors]\n        representing anchorwise classification losses.\n      decoded_boxlist_list: a list of decoded BoxList representing location\n        predictions for each image.\n      match_list: an optional list of matcher.Match objects encoding the match\n        between anchors and groundtruth boxes for each image of the batch,\n        with rows of the Match objects corresponding to groundtruth boxes\n        and columns corresponding to anchors.  Match objects in match_list are\n        used to reference which anchors are positive, negative or ignored.  If\n        self._max_negatives_per_positive exists, these are then used to enforce\n        a prespecified negative to positive ratio.\n\n    Returns:\n      mined_location_loss: a float scalar with sum of localization losses from\n        selected hard examples.\n      mined_cls_loss: a float scalar with sum of classification losses from\n        selected hard examples.\n    Raises:\n      ValueError: if location_losses, cls_losses and decoded_boxlist_list do\n        not have compatible shapes (i.e., they must correspond to the same\n        number of images).\n      ValueError: if match_list is specified but its length does not match\n        len(decoded_boxlist_list).\n    '
        mined_location_losses = []
        mined_cls_losses = []
        location_losses = tf.unstack(location_losses)
        cls_losses = tf.unstack(cls_losses)
        num_images = len(decoded_boxlist_list)
        if (not match_list):
            match_list = (num_images * [None])
        if (not (len(location_losses) == len(decoded_boxlist_list) == len(cls_losses))):
            raise ValueError('location_losses, cls_losses and decoded_boxlist_list do not have compatible shapes.')
        if (not isinstance(match_list, list)):
            raise ValueError('match_list must be a list.')
        if (len(match_list) != len(decoded_boxlist_list)):
            raise ValueError('match_list must either be None or have length=len(decoded_boxlist_list).')
        num_positives_list = []
        num_negatives_list = []
        for (ind, detection_boxlist) in enumerate(decoded_boxlist_list):
            box_locations = detection_boxlist.get()
            match = match_list[ind]
            image_losses = cls_losses[ind]
            if (self._loss_type == 'loc'):
                image_losses = location_losses[ind]
            elif (self._loss_type == 'both'):
                image_losses *= self._cls_loss_weight
                image_losses += (location_losses[ind] * self._loc_loss_weight)
            if (self._num_hard_examples is not None):
                num_hard_examples = self._num_hard_examples
            else:
                num_hard_examples = detection_boxlist.num_boxes()
            with tf.device('/cpu:0'):
                selected_indices = tf.image.non_max_suppression(box_locations, image_losses, num_hard_examples, self._iou_threshold)
            if ((self._max_negatives_per_positive is not None) and match):
                (selected_indices, num_positives, num_negatives) = self._subsample_selection_to_desired_neg_pos_ratio(selected_indices, match, self._max_negatives_per_positive, self._min_negatives_per_image)
                num_positives_list.append(num_positives)
                num_negatives_list.append(num_negatives)
            mined_location_losses.append(tf.reduce_sum(tf.gather(location_losses[ind], selected_indices)))
            mined_cls_losses.append(tf.reduce_sum(tf.gather(cls_losses[ind], selected_indices)))
        location_loss = tf.reduce_sum(tf.stack(mined_location_losses))
        cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses))
        if (match and self._max_negatives_per_positive):
            self._num_positives_list = num_positives_list
            self._num_negatives_list = num_negatives_list
        return (location_loss, cls_loss)

    def summarize(self):
        'Summarize the number of positives and negatives after mining.'
        if (self._num_positives_list and self._num_negatives_list):
            avg_num_positives = tf.reduce_mean(tf.to_float(self._num_positives_list))
            avg_num_negatives = tf.reduce_mean(tf.to_float(self._num_negatives_list))
            tf.summary.scalar('HardExampleMiner/NumPositives', avg_num_positives)
            tf.summary.scalar('HardExampleMiner/NumNegatives', avg_num_negatives)

    def _subsample_selection_to_desired_neg_pos_ratio(self, indices, match, max_negatives_per_positive, min_negatives_per_image=0):
        "Subsample a collection of selected indices to a desired neg:pos ratio.\n\n    This function takes a subset of M indices (indexing into a large anchor\n    collection of N anchors where M<N) which are labeled as positive/negative\n    via a Match object (matched indices are positive, unmatched indices\n    are negative).  It returns a subset of the provided indices retaining all\n    positives as well as up to the first K negatives, where:\n      K=floor(num_negative_per_positive * num_positives).\n\n    For example, if indices=[2, 4, 5, 7, 9, 10] (indexing into 12 anchors),\n    with positives=[2, 5] and negatives=[4, 7, 9, 10] and\n    num_negatives_per_positive=1, then the returned subset of indices\n    is [2, 4, 5, 7].\n\n    Args:\n      indices: An integer tensor of shape [M] representing a collection\n        of selected anchor indices\n      match: A matcher.Match object encoding the match between anchors and\n        groundtruth boxes for a given image, with rows of the Match objects\n        corresponding to groundtruth boxes and columns corresponding to anchors.\n      max_negatives_per_positive: (float) maximum number of negatives for\n        each positive anchor.\n      min_negatives_per_image: minimum number of negative anchors for a given\n        image. Allow sampling negatives in image without any positive anchors.\n\n    Returns:\n      selected_indices: An integer tensor of shape [M'] representing a\n        collection of selected anchor indices with M' <= M.\n      num_positives: An integer tensor representing the number of positive\n        examples in selected set of indices.\n      num_negatives: An integer tensor representing the number of negative\n        examples in selected set of indices.\n    "
        positives_indicator = tf.gather(match.matched_column_indicator(), indices)
        negatives_indicator = tf.gather(match.unmatched_column_indicator(), indices)
        num_positives = tf.reduce_sum(tf.to_int32(positives_indicator))
        max_negatives = tf.maximum(min_negatives_per_image, tf.to_int32((max_negatives_per_positive * tf.to_float(num_positives))))
        topk_negatives_indicator = tf.less_equal(tf.cumsum(tf.to_int32(negatives_indicator)), max_negatives)
        subsampled_selection_indices = tf.where(tf.logical_or(positives_indicator, topk_negatives_indicator))
        num_negatives = (tf.size(subsampled_selection_indices) - num_positives)
        return (tf.reshape(tf.gather(indices, subsampled_selection_indices), [(- 1)]), num_positives, num_negatives)
