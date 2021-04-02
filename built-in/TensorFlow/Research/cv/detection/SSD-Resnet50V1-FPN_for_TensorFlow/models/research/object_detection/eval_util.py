
'Common utility functions for evaluation.'
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import collections
import os
import time
import numpy as np
import tensorflow as tf
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.core import standard_fields as fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils as vis_utils

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config
slim = tf.contrib.slim
EVAL_METRICS_CLASS_DICT = {'coco_detection_metrics': coco_evaluation.CocoDetectionEvaluator, 'coco_mask_metrics': coco_evaluation.CocoMaskEvaluator, 'oid_challenge_detection_metrics': object_detection_evaluation.OpenImagesDetectionChallengeEvaluator, 'pascal_voc_detection_metrics': object_detection_evaluation.PascalDetectionEvaluator, 'weighted_pascal_voc_detection_metrics': object_detection_evaluation.WeightedPascalDetectionEvaluator, 'pascal_voc_instance_segmentation_metrics': object_detection_evaluation.PascalInstanceSegmentationEvaluator, 'weighted_pascal_voc_instance_segmentation_metrics': object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator, 'oid_V2_detection_metrics': object_detection_evaluation.OpenImagesDetectionEvaluator}
EVAL_DEFAULT_METRIC = 'coco_detection_metrics'

def write_metrics(metrics, global_step, summary_dir):
    'Write metrics to a summary directory.\n\n  Args:\n    metrics: A dictionary containing metric names and values.\n    global_step: Global step at which the metrics are computed.\n    summary_dir: Directory to write tensorflow summaries to.\n  '
    tf.logging.info('Writing metrics to tf summary.')
    summary_writer = tf.summary.FileWriterCache.get(summary_dir)
    for key in sorted(metrics):
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=metrics[key])])
        summary_writer.add_summary(summary, global_step)
        tf.logging.info('%s: %f', key, metrics[key])
    tf.logging.info('Metrics written to tf summary.')

def visualize_detection_results(result_dict, tag, global_step, categories, summary_dir='', export_dir='', agnostic_mode=False, show_groundtruth=False, groundtruth_box_visualization_color='black', min_score_thresh=0.5, max_num_predictions=20, skip_scores=False, skip_labels=False, keep_image_id_for_visualization_export=False):
    'Visualizes detection results and writes visualizations to image summaries.\n\n  This function visualizes an image with its detected bounding boxes and writes\n  to image summaries which can be viewed on tensorboard.  It optionally also\n  writes images to a directory. In the case of missing entry in the label map,\n  unknown class name in the visualization is shown as "N/A".\n\n  Args:\n    result_dict: a dictionary holding groundtruth and detection\n      data corresponding to each image being evaluated.  The following keys\n      are required:\n        \'original_image\': a numpy array representing the image with shape\n          [1, height, width, 3] or [1, height, width, 1]\n        \'detection_boxes\': a numpy array of shape [N, 4]\n        \'detection_scores\': a numpy array of shape [N]\n        \'detection_classes\': a numpy array of shape [N]\n      The following keys are optional:\n        \'groundtruth_boxes\': a numpy array of shape [N, 4]\n        \'groundtruth_keypoints\': a numpy array of shape [N, num_keypoints, 2]\n      Detections are assumed to be provided in decreasing order of score and for\n      display, and we assume that scores are probabilities between 0 and 1.\n    tag: tensorboard tag (string) to associate with image.\n    global_step: global step at which the visualization are generated.\n    categories: a list of dictionaries representing all possible categories.\n      Each dict in this list has the following keys:\n          \'id\': (required) an integer id uniquely identifying this category\n          \'name\': (required) string representing category name\n            e.g., \'cat\', \'dog\', \'pizza\'\n          \'supercategory\': (optional) string representing the supercategory\n            e.g., \'animal\', \'vehicle\', \'food\', etc\n    summary_dir: the output directory to which the image summaries are written.\n    export_dir: the output directory to which images are written.  If this is\n      empty (default), then images are not exported.\n    agnostic_mode: boolean (default: False) controlling whether to evaluate in\n      class-agnostic mode or not.\n    show_groundtruth: boolean (default: False) controlling whether to show\n      groundtruth boxes in addition to detected boxes\n    groundtruth_box_visualization_color: box color for visualizing groundtruth\n      boxes\n    min_score_thresh: minimum score threshold for a box to be visualized\n    max_num_predictions: maximum number of detections to visualize\n    skip_scores: whether to skip score when drawing a single detection\n    skip_labels: whether to skip label when drawing a single detection\n    keep_image_id_for_visualization_export: whether to keep image identifier in\n      filename when exported to export_dir\n  Raises:\n    ValueError: if result_dict does not contain the expected keys (i.e.,\n      \'original_image\', \'detection_boxes\', \'detection_scores\',\n      \'detection_classes\')\n  '
    detection_fields = fields.DetectionResultFields
    input_fields = fields.InputDataFields
    if (not set([input_fields.original_image, detection_fields.detection_boxes, detection_fields.detection_scores, detection_fields.detection_classes]).issubset(set(result_dict.keys()))):
        raise ValueError('result_dict does not contain all expected keys.')
    if (show_groundtruth and (input_fields.groundtruth_boxes not in result_dict)):
        raise ValueError('If show_groundtruth is enabled, result_dict must contain groundtruth_boxes.')
    tf.logging.info('Creating detection visualizations.')
    category_index = label_map_util.create_category_index(categories)
    image = np.squeeze(result_dict[input_fields.original_image], axis=0)
    if (image.shape[2] == 1):
        image = np.tile(image, [1, 1, 3])
    detection_boxes = result_dict[detection_fields.detection_boxes]
    detection_scores = result_dict[detection_fields.detection_scores]
    detection_classes = np.int32(result_dict[detection_fields.detection_classes])
    detection_keypoints = result_dict.get(detection_fields.detection_keypoints)
    detection_masks = result_dict.get(detection_fields.detection_masks)
    detection_boundaries = result_dict.get(detection_fields.detection_boundaries)
    if show_groundtruth:
        groundtruth_boxes = result_dict[input_fields.groundtruth_boxes]
        groundtruth_keypoints = result_dict.get(input_fields.groundtruth_keypoints)
        vis_utils.visualize_boxes_and_labels_on_image_array(image=image, boxes=groundtruth_boxes, classes=None, scores=None, category_index=category_index, keypoints=groundtruth_keypoints, use_normalized_coordinates=False, max_boxes_to_draw=None, groundtruth_box_visualization_color=groundtruth_box_visualization_color)
    vis_utils.visualize_boxes_and_labels_on_image_array(image, detection_boxes, detection_classes, detection_scores, category_index, instance_masks=detection_masks, instance_boundaries=detection_boundaries, keypoints=detection_keypoints, use_normalized_coordinates=False, max_boxes_to_draw=max_num_predictions, min_score_thresh=min_score_thresh, agnostic_mode=agnostic_mode, skip_scores=skip_scores, skip_labels=skip_labels)
    if export_dir:
        if (keep_image_id_for_visualization_export and result_dict[fields.InputDataFields().key]):
            export_path = os.path.join(export_dir, 'export-{}-{}.png'.format(tag, result_dict[fields.InputDataFields().key]))
        else:
            export_path = os.path.join(export_dir, 'export-{}.png'.format(tag))
        vis_utils.save_image_array_as_png(image, export_path)
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=tf.Summary.Image(encoded_image_string=vis_utils.encode_image_array_as_png_str(image)))])
    summary_writer = tf.summary.FileWriterCache.get(summary_dir)
    summary_writer.add_summary(summary, global_step)
    tf.logging.info('Detection visualizations written to summary with tag %s.', tag)

def _run_checkpoint_once(tensor_dict, evaluators=None, batch_processor=None, checkpoint_dirs=None, variables_to_restore=None, restore_fn=None, num_batches=1, master='', save_graph=False, save_graph_dir='', losses_dict=None, eval_export_path=None):
    "Evaluates metrics defined in evaluators and returns summaries.\n\n  This function loads the latest checkpoint in checkpoint_dirs and evaluates\n  all metrics defined in evaluators. The metrics are processed in batch by the\n  batch_processor.\n\n  Args:\n    tensor_dict: a dictionary holding tensors representing a batch of detections\n      and corresponding groundtruth annotations.\n    evaluators: a list of object of type DetectionEvaluator to be used for\n      evaluation. Note that the metric names produced by different evaluators\n      must be unique.\n    batch_processor: a function taking four arguments:\n      1. tensor_dict: the same tensor_dict that is passed in as the first\n        argument to this function.\n      2. sess: a tensorflow session\n      3. batch_index: an integer representing the index of the batch amongst\n        all batches\n      By default, batch_processor is None, which defaults to running:\n        return sess.run(tensor_dict)\n      To skip an image, it suffices to return an empty dictionary in place of\n      result_dict.\n    checkpoint_dirs: list of directories to load into an EnsembleModel. If it\n      has only one directory, EnsembleModel will not be used --\n        a DetectionModel\n      will be instantiated directly. Not used if restore_fn is set.\n    variables_to_restore: None, or a dictionary mapping variable names found in\n      a checkpoint to model variables. The dictionary would normally be\n      generated by creating a tf.train.ExponentialMovingAverage object and\n      calling its variables_to_restore() method. Not used if restore_fn is set.\n    restore_fn: None, or a function that takes a tf.Session object and correctly\n      restores all necessary variables from the correct checkpoint file. If\n      None, attempts to restore from the first directory in checkpoint_dirs.\n    num_batches: the number of batches to use for evaluation.\n    master: the location of the Tensorflow session.\n    save_graph: whether or not the Tensorflow graph is stored as a pbtxt file.\n    save_graph_dir: where to store the Tensorflow graph on disk. If save_graph\n      is True this must be non-empty.\n    losses_dict: optional dictionary of scalar detection losses.\n    eval_export_path: Path for saving a json file that contains the detection\n      results in json format.\n\n  Returns:\n    global_step: the count of global steps.\n    all_evaluator_metrics: A dictionary containing metric names and values.\n\n  Raises:\n    ValueError: if restore_fn is None and checkpoint_dirs doesn't have at least\n      one element.\n    ValueError: if save_graph is True and save_graph_dir is not defined.\n  "
    if (save_graph and (not save_graph_dir)):
        raise ValueError('`save_graph_dir` must be defined.')
    sess = tf.Session(master, graph=tf.get_default_graph(), config=npu_session_config_init())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    if restore_fn:
        restore_fn(sess)
    else:
        if (not checkpoint_dirs):
            raise ValueError('`checkpoint_dirs` must have at least one entry.')
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dirs[0])
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, checkpoint_file)
    if save_graph:
        tf.train.write_graph(sess.graph_def, save_graph_dir, 'eval.pbtxt')
    counters = {'skipped': 0, 'success': 0}
    aggregate_result_losses_dict = collections.defaultdict(list)
    with tf.contrib.slim.queues.QueueRunners(sess):
        try:
            for batch in range(int(num_batches)):
                if (((batch + 1) % 100) == 0):
                    tf.logging.info('Running eval ops batch %d/%d', (batch + 1), num_batches)
                if (not batch_processor):
                    try:
                        if (not losses_dict):
                            losses_dict = {}
                        (result_dict, result_losses_dict) = sess.run([tensor_dict, losses_dict])
                        counters['success'] += 1
                    except tf.errors.InvalidArgumentError:
                        tf.logging.info('Skipping image')
                        counters['skipped'] += 1
                        result_dict = {}
                else:
                    (result_dict, result_losses_dict) = batch_processor(tensor_dict, sess, batch, counters, losses_dict=losses_dict)
                if (not result_dict):
                    continue
                for (key, value) in iter(result_losses_dict.items()):
                    aggregate_result_losses_dict[key].append(value)
                for evaluator in evaluators:
                    if (isinstance(result_dict, dict) and (fields.InputDataFields.key in result_dict) and result_dict[fields.InputDataFields.key]):
                        image_id = result_dict[fields.InputDataFields.key]
                    else:
                        image_id = batch
                    evaluator.add_single_ground_truth_image_info(image_id=image_id, groundtruth_dict=result_dict)
                    evaluator.add_single_detected_image_info(image_id=image_id, detections_dict=result_dict)
            tf.logging.info('Running eval batches done.')
        except tf.errors.OutOfRangeError:
            tf.logging.info('Done evaluating -- epoch limit reached')
        finally:
            tf.logging.info('# success: %d', counters['success'])
            tf.logging.info('# skipped: %d', counters['skipped'])
            all_evaluator_metrics = {}
            if (eval_export_path and (eval_export_path is not None)):
                for evaluator in evaluators:
                    if (isinstance(evaluator, coco_evaluation.CocoDetectionEvaluator) or isinstance(evaluator, coco_evaluation.CocoMaskEvaluator)):
                        tf.logging.info('Started dumping to json file.')
                        evaluator.dump_detections_to_json_file(json_output_path=eval_export_path)
                        tf.logging.info('Finished dumping to json file.')
            for evaluator in evaluators:
                metrics = evaluator.evaluate()
                evaluator.clear()
                if any(((key in all_evaluator_metrics) for key in metrics)):
                    raise ValueError('Metric names between evaluators must not collide.')
                all_evaluator_metrics.update(metrics)
            global_step = tf.train.global_step(sess, tf.train.get_global_step())
            for (key, value) in iter(aggregate_result_losses_dict.items()):
                all_evaluator_metrics[('Losses/' + key)] = np.mean(value)
    sess.close()
    return (global_step, all_evaluator_metrics)

def repeated_checkpoint_run(tensor_dict, summary_dir, evaluators, batch_processor=None, checkpoint_dirs=None, variables_to_restore=None, restore_fn=None, num_batches=1, eval_interval_secs=120, max_number_of_evaluations=None, master='', save_graph=False, save_graph_dir='', losses_dict=None, eval_export_path=None):
    "Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.\n\n  This function repeatedly loads a checkpoint and evaluates a desired\n  set of tensors (provided by tensor_dict) and hands the resulting numpy\n  arrays to a function result_processor which can be used to further\n  process/save/visualize the results.\n\n  Args:\n    tensor_dict: a dictionary holding tensors representing a batch of detections\n      and corresponding groundtruth annotations.\n    summary_dir: a directory to write metrics summaries.\n    evaluators: a list of object of type DetectionEvaluator to be used for\n      evaluation. Note that the metric names produced by different evaluators\n      must be unique.\n    batch_processor: a function taking three arguments:\n      1. tensor_dict: the same tensor_dict that is passed in as the first\n        argument to this function.\n      2. sess: a tensorflow session\n      3. batch_index: an integer representing the index of the batch amongst\n        all batches\n      By default, batch_processor is None, which defaults to running:\n        return sess.run(tensor_dict)\n    checkpoint_dirs: list of directories to load into a DetectionModel or an\n      EnsembleModel if restore_fn isn't set. Also used to determine when to run\n      next evaluation. Must have at least one element.\n    variables_to_restore: None, or a dictionary mapping variable names found in\n      a checkpoint to model variables. The dictionary would normally be\n      generated by creating a tf.train.ExponentialMovingAverage object and\n      calling its variables_to_restore() method. Not used if restore_fn is set.\n    restore_fn: a function that takes a tf.Session object and correctly restores\n      all necessary variables from the correct checkpoint file.\n    num_batches: the number of batches to use for evaluation.\n    eval_interval_secs: the number of seconds between each evaluation run.\n    max_number_of_evaluations: the max number of iterations of the evaluation.\n      If the value is left as None the evaluation continues indefinitely.\n    master: the location of the Tensorflow session.\n    save_graph: whether or not the Tensorflow graph is saved as a pbtxt file.\n    save_graph_dir: where to save on disk the Tensorflow graph. If store_graph\n      is True this must be non-empty.\n    losses_dict: optional dictionary of scalar detection losses.\n    eval_export_path: Path for saving a json file that contains the detection\n      results in json format.\n\n  Returns:\n    metrics: A dictionary containing metric names and values in the latest\n      evaluation.\n\n  Raises:\n    ValueError: if max_num_of_evaluations is not None or a positive number.\n    ValueError: if checkpoint_dirs doesn't have at least one element.\n  "
    if (max_number_of_evaluations and (max_number_of_evaluations <= 0)):
        raise ValueError('`number_of_steps` must be either None or a positive number.')
    if (not checkpoint_dirs):
        raise ValueError('`checkpoint_dirs` must have at least one entry.')
    last_evaluated_model_path = None
    number_of_evaluations = 0
    while True:
        start = time.time()
        tf.logging.info(('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())))
        model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        if (not model_path):
            tf.logging.info('No model found in %s. Will try again in %d seconds', checkpoint_dirs[0], eval_interval_secs)
        elif (model_path == last_evaluated_model_path):
            tf.logging.info('Found already evaluated checkpoint. Will try again in %d seconds', eval_interval_secs)
        else:
            last_evaluated_model_path = model_path
            (global_step, metrics) = _run_checkpoint_once(tensor_dict, evaluators, batch_processor, checkpoint_dirs, variables_to_restore, restore_fn, num_batches, master, save_graph, save_graph_dir, losses_dict=losses_dict, eval_export_path=eval_export_path)
            write_metrics(metrics, global_step, summary_dir)
        number_of_evaluations += 1
        if (max_number_of_evaluations and (number_of_evaluations >= max_number_of_evaluations)):
            tf.logging.info('Finished evaluation!')
            break
        time_to_next_eval = ((start + eval_interval_secs) - time.time())
        if (time_to_next_eval > 0):
            time.sleep(time_to_next_eval)
    return metrics

def _scale_box_to_absolute(args):
    (boxes, image_shape) = args
    return box_list_ops.to_absolute_coordinates(box_list.BoxList(boxes), image_shape[0], image_shape[1]).get()

def _resize_detection_masks(args):
    (detection_boxes, detection_masks, image_shape) = args
    detection_masks_reframed = ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_shape[0], image_shape[1])
    return tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

def _resize_groundtruth_masks(args):
    (mask, image_shape) = args
    mask = tf.expand_dims(mask, 3)
    mask = tf.image.resize_images(mask, image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    return tf.cast(tf.squeeze(mask, 3), tf.uint8)

def _scale_keypoint_to_absolute(args):
    (keypoints, image_shape) = args
    return keypoint_ops.scale(keypoints, image_shape[0], image_shape[1])

def result_dict_for_single_example(image, key, detections, groundtruth=None, class_agnostic=False, scale_to_absolute=False):
    "Merges all detection and groundtruth information for a single example.\n\n  Note that evaluation tools require classes that are 1-indexed, and so this\n  function performs the offset. If `class_agnostic` is True, all output classes\n  have label 1.\n\n  Args:\n    image: A single 4D uint8 image tensor of shape [1, H, W, C].\n    key: A single string tensor identifying the image.\n    detections: A dictionary of detections, returned from\n      DetectionModel.postprocess().\n    groundtruth: (Optional) Dictionary of groundtruth items, with fields:\n      'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in\n        normalized coordinates.\n      'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.\n      'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)\n      'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)\n      'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)\n      'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)\n      'groundtruth_instance_masks': 3D int64 tensor of instance masks\n        (Optional).\n    class_agnostic: Boolean indicating whether the detections are class-agnostic\n      (i.e. binary). Default False.\n    scale_to_absolute: Boolean indicating whether boxes and keypoints should be\n      scaled to absolute coordinates. Note that for IoU based evaluations, it\n      does not matter whether boxes are expressed in absolute or relative\n      coordinates. Default False.\n\n  Returns:\n    A dictionary with:\n    'original_image': A [1, H, W, C] uint8 image tensor.\n    'key': A string tensor with image identifier.\n    'detection_boxes': [max_detections, 4] float32 tensor of boxes, in\n      normalized or absolute coordinates, depending on the value of\n      `scale_to_absolute`.\n    'detection_scores': [max_detections] float32 tensor of scores.\n    'detection_classes': [max_detections] int64 tensor of 1-indexed classes.\n    'detection_masks': [max_detections, H, W] float32 tensor of binarized\n      masks, reframed to full image masks.\n    'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in\n      normalized or absolute coordinates, depending on the value of\n      `scale_to_absolute`. (Optional)\n    'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.\n      (Optional)\n    'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)\n    'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)\n    'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)\n    'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)\n    'groundtruth_instance_masks': 3D int64 tensor of instance masks\n      (Optional).\n\n  "
    if groundtruth:
        max_gt_boxes = tf.shape(groundtruth[fields.InputDataFields.groundtruth_boxes])[0]
        for gt_key in groundtruth:
            groundtruth[gt_key] = tf.expand_dims(groundtruth[gt_key], 0)
    for detection_key in detections:
        detections[detection_key] = tf.expand_dims(detections[detection_key][0], axis=0)
    batched_output_dict = result_dict_for_batched_example(image, tf.expand_dims(key, 0), detections, groundtruth, class_agnostic, scale_to_absolute, max_gt_boxes=max_gt_boxes)
    exclude_keys = [fields.InputDataFields.original_image, fields.DetectionResultFields.num_detections, fields.InputDataFields.num_groundtruth_boxes]
    output_dict = {fields.InputDataFields.original_image: batched_output_dict[fields.InputDataFields.original_image]}
    for key in batched_output_dict:
        if (key not in exclude_keys):
            output_dict[key] = tf.squeeze(batched_output_dict[key], 0)
    return output_dict

def result_dict_for_batched_example(images, keys, detections, groundtruth=None, class_agnostic=False, scale_to_absolute=False, original_image_spatial_shapes=None, true_image_shapes=None, max_gt_boxes=None):
    "Merges all detection and groundtruth information for a single example.\n\n  Note that evaluation tools require classes that are 1-indexed, and so this\n  function performs the offset. If `class_agnostic` is True, all output classes\n  have label 1.\n\n  Args:\n    images: A single 4D uint8 image tensor of shape [batch_size, H, W, C].\n    keys: A [batch_size] string tensor with image identifier.\n    detections: A dictionary of detections, returned from\n      DetectionModel.postprocess().\n    groundtruth: (Optional) Dictionary of groundtruth items, with fields:\n      'groundtruth_boxes': [batch_size, max_number_of_boxes, 4] float32 tensor\n        of boxes, in normalized coordinates.\n      'groundtruth_classes':  [batch_size, max_number_of_boxes] int64 tensor of\n        1-indexed classes.\n      'groundtruth_area': [batch_size, max_number_of_boxes] float32 tensor of\n        bbox area. (Optional)\n      'groundtruth_is_crowd':[batch_size, max_number_of_boxes] int64\n        tensor. (Optional)\n      'groundtruth_difficult': [batch_size, max_number_of_boxes] int64\n        tensor. (Optional)\n      'groundtruth_group_of': [batch_size, max_number_of_boxes] int64\n        tensor. (Optional)\n      'groundtruth_instance_masks': 4D int64 tensor of instance\n        masks (Optional).\n    class_agnostic: Boolean indicating whether the detections are class-agnostic\n      (i.e. binary). Default False.\n    scale_to_absolute: Boolean indicating whether boxes and keypoints should be\n      scaled to absolute coordinates. Note that for IoU based evaluations, it\n      does not matter whether boxes are expressed in absolute or relative\n      coordinates. Default False.\n    original_image_spatial_shapes: A 2D int32 tensor of shape [batch_size, 2]\n      used to resize the image. When set to None, the image size is retained.\n    true_image_shapes: A 2D int32 tensor of shape [batch_size, 3]\n      containing the size of the unpadded original_image.\n    max_gt_boxes: [batch_size] tensor representing the maximum number of\n      groundtruth boxes to pad.\n\n  Returns:\n    A dictionary with:\n    'original_image': A [batch_size, H, W, C] uint8 image tensor.\n    'original_image_spatial_shape': A [batch_size, 2] tensor containing the\n      original image sizes.\n    'true_image_shape': A [batch_size, 3] tensor containing the size of\n      the unpadded original_image.\n    'key': A [batch_size] string tensor with image identifier.\n    'detection_boxes': [batch_size, max_detections, 4] float32 tensor of boxes,\n      in normalized or absolute coordinates, depending on the value of\n      `scale_to_absolute`.\n    'detection_scores': [batch_size, max_detections] float32 tensor of scores.\n    'detection_classes': [batch_size, max_detections] int64 tensor of 1-indexed\n      classes.\n    'detection_masks': [batch_size, max_detections, H, W] float32 tensor of\n      binarized masks, reframed to full image masks.\n    'num_detections': [batch_size] int64 tensor containing number of valid\n      detections.\n    'groundtruth_boxes': [batch_size, num_boxes, 4] float32 tensor of boxes, in\n      normalized or absolute coordinates, depending on the value of\n      `scale_to_absolute`. (Optional)\n    'groundtruth_classes': [batch_size, num_boxes] int64 tensor of 1-indexed\n      classes. (Optional)\n    'groundtruth_area': [batch_size, num_boxes] float32 tensor of bbox\n      area. (Optional)\n    'groundtruth_is_crowd': [batch_size, num_boxes] int64 tensor. (Optional)\n    'groundtruth_difficult': [batch_size, num_boxes] int64 tensor. (Optional)\n    'groundtruth_group_of': [batch_size, num_boxes] int64 tensor. (Optional)\n    'groundtruth_instance_masks': 4D int64 tensor of instance masks\n      (Optional).\n    'num_groundtruth_boxes': [batch_size] tensor containing the maximum number\n      of groundtruth boxes per image.\n\n  Raises:\n    ValueError: if original_image_spatial_shape is not 2D int32 tensor of shape\n      [2].\n    ValueError: if true_image_shapes is not 2D int32 tensor of shape\n      [3].\n  "
    label_id_offset = 1
    input_data_fields = fields.InputDataFields
    if (original_image_spatial_shapes is None):
        original_image_spatial_shapes = tf.tile(tf.expand_dims(tf.shape(images)[1:3], axis=0), multiples=[tf.shape(images)[0], 1])
    elif ((len(original_image_spatial_shapes.shape) != 2) and (original_image_spatial_shapes.shape[1] != 2)):
        raise ValueError('`original_image_spatial_shape` should be a 2D tensor of shape [batch_size, 2].')
    if (true_image_shapes is None):
        true_image_shapes = tf.tile(tf.expand_dims(tf.shape(images)[1:4], axis=0), multiples=[tf.shape(images)[0], 1])
    elif ((len(true_image_shapes.shape) != 2) and (true_image_shapes.shape[1] != 3)):
        raise ValueError('`true_image_shapes` should be a 2D tensor of shape [batch_size, 3].')
    output_dict = {input_data_fields.original_image: images, input_data_fields.key: keys, input_data_fields.original_image_spatial_shape: original_image_spatial_shapes, input_data_fields.true_image_shape: true_image_shapes}
    detection_fields = fields.DetectionResultFields
    detection_boxes = detections[detection_fields.detection_boxes]
    detection_scores = detections[detection_fields.detection_scores]
    num_detections = tf.to_int32(detections[detection_fields.num_detections])
    if class_agnostic:
        detection_classes = tf.ones_like(detection_scores, dtype=tf.int64)
    else:
        detection_classes = (tf.to_int64(detections[detection_fields.detection_classes]) + label_id_offset)
    if scale_to_absolute:
        output_dict[detection_fields.detection_boxes] = shape_utils.static_or_dynamic_map_fn(_scale_box_to_absolute, elems=[detection_boxes, original_image_spatial_shapes], dtype=tf.float32)
    else:
        output_dict[detection_fields.detection_boxes] = detection_boxes
    output_dict[detection_fields.detection_classes] = detection_classes
    output_dict[detection_fields.detection_scores] = detection_scores
    output_dict[detection_fields.num_detections] = num_detections
    if (detection_fields.detection_masks in detections):
        detection_masks = detections[detection_fields.detection_masks]
        output_dict[detection_fields.detection_masks] = shape_utils.static_or_dynamic_map_fn(_resize_detection_masks, elems=[detection_boxes, detection_masks, original_image_spatial_shapes], dtype=tf.uint8)
    if (detection_fields.detection_keypoints in detections):
        detection_keypoints = detections[detection_fields.detection_keypoints]
        output_dict[detection_fields.detection_keypoints] = detection_keypoints
        if scale_to_absolute:
            output_dict[detection_fields.detection_keypoints] = shape_utils.static_or_dynamic_map_fn(_scale_keypoint_to_absolute, elems=[detection_keypoints, original_image_spatial_shapes], dtype=tf.float32)
    if groundtruth:
        if (max_gt_boxes is None):
            if (input_data_fields.num_groundtruth_boxes in groundtruth):
                max_gt_boxes = groundtruth[input_data_fields.num_groundtruth_boxes]
            else:
                raise ValueError('max_gt_boxes must be provided when processing batched examples.')
        if (input_data_fields.groundtruth_instance_masks in groundtruth):
            masks = groundtruth[input_data_fields.groundtruth_instance_masks]
            groundtruth[input_data_fields.groundtruth_instance_masks] = shape_utils.static_or_dynamic_map_fn(_resize_groundtruth_masks, elems=[masks, original_image_spatial_shapes], dtype=tf.uint8)
        output_dict.update(groundtruth)
        if scale_to_absolute:
            groundtruth_boxes = groundtruth[input_data_fields.groundtruth_boxes]
            output_dict[input_data_fields.groundtruth_boxes] = shape_utils.static_or_dynamic_map_fn(_scale_box_to_absolute, elems=[groundtruth_boxes, original_image_spatial_shapes], dtype=tf.float32)
        if class_agnostic:
            groundtruth_classes = groundtruth[input_data_fields.groundtruth_classes]
            groundtruth_classes = tf.ones_like(groundtruth_classes, dtype=tf.int64)
            output_dict[input_data_fields.groundtruth_classes] = groundtruth_classes
        output_dict[input_data_fields.num_groundtruth_boxes] = max_gt_boxes
    return output_dict

def get_evaluators(eval_config, categories, evaluator_options=None):
    "Returns the evaluator class according to eval_config, valid for categories.\n\n  Args:\n    eval_config: An `eval_pb2.EvalConfig`.\n    categories: A list of dicts, each of which has the following keys -\n        'id': (required) an integer id uniquely identifying this category.\n        'name': (required) string representing category name e.g., 'cat', 'dog'.\n    evaluator_options: A dictionary of metric names (see\n      EVAL_METRICS_CLASS_DICT) to `DetectionEvaluator` initialization\n      keyword arguments. For example:\n      evalator_options = {\n        'coco_detection_metrics': {'include_metrics_per_category': True}\n      }\n\n  Returns:\n    An list of instances of DetectionEvaluator.\n\n  Raises:\n    ValueError: if metric is not in the metric class dictionary.\n  "
    evaluator_options = (evaluator_options or {})
    eval_metric_fn_keys = eval_config.metrics_set
    if (not eval_metric_fn_keys):
        eval_metric_fn_keys = [EVAL_DEFAULT_METRIC]
    evaluators_list = []
    for eval_metric_fn_key in eval_metric_fn_keys:
        if (eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT):
            raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
        kwargs_dict = (evaluator_options[eval_metric_fn_key] if (eval_metric_fn_key in evaluator_options) else {})
        evaluators_list.append(EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories, **kwargs_dict))
    return evaluators_list

def get_eval_metric_ops_for_evaluators(eval_config, categories, eval_dict):
    "Returns eval metrics ops to use with `tf.estimator.EstimatorSpec`.\n\n  Args:\n    eval_config: An `eval_pb2.EvalConfig`.\n    categories: A list of dicts, each of which has the following keys -\n        'id': (required) an integer id uniquely identifying this category.\n        'name': (required) string representing category name e.g., 'cat', 'dog'.\n    eval_dict: An evaluation dictionary, returned from\n      result_dict_for_single_example().\n\n  Returns:\n    A dictionary of metric names to tuple of value_op and update_op that can be\n    used as eval metric ops in tf.EstimatorSpec.\n  "
    eval_metric_ops = {}
    evaluator_options = evaluator_options_from_eval_config(eval_config)
    evaluators_list = get_evaluators(eval_config, categories, evaluator_options)
    for evaluator in evaluators_list:
        eval_metric_ops.update(evaluator.get_estimator_eval_metric_ops(eval_dict))
    return eval_metric_ops

def evaluator_options_from_eval_config(eval_config):
    "Produces a dictionary of evaluation options for each eval metric.\n\n  Args:\n    eval_config: An `eval_pb2.EvalConfig`.\n\n  Returns:\n    evaluator_options: A dictionary of metric names (see\n      EVAL_METRICS_CLASS_DICT) to `DetectionEvaluator` initialization\n      keyword arguments. For example:\n      evalator_options = {\n        'coco_detection_metrics': {'include_metrics_per_category': True}\n      }\n  "
    eval_metric_fn_keys = eval_config.metrics_set
    evaluator_options = {}
    for eval_metric_fn_key in eval_metric_fn_keys:
        if (eval_metric_fn_key in ('coco_detection_metrics', 'coco_mask_metrics')):
            evaluator_options[eval_metric_fn_key] = {'include_metrics_per_category': eval_config.include_metrics_per_category}
    return evaluator_options
