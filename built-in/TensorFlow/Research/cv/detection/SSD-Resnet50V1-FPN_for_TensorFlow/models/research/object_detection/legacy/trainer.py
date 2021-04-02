
'Detection model trainer.\n\nThis file provides a generic training method that can be used to train a\nDetectionModel.\n'
from npu_bridge.npu_init import *
from tensorflow.core.protobuf import config_pb2
import functools
import tensorflow as tf
from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt
slim = tf.contrib.slim

def create_input_queue(batch_size_per_clone, create_tensor_dict_fn, batch_queue_capacity, num_batch_queue_threads, prefetch_queue_capacity, data_augmentation_options):
    'Sets up reader, prefetcher and returns input queue.\n\n  Args:\n    batch_size_per_clone: batch size to use per clone.\n    create_tensor_dict_fn: function to create tensor dictionary.\n    batch_queue_capacity: maximum number of elements to store within a queue.\n    num_batch_queue_threads: number of threads to use for batching.\n    prefetch_queue_capacity: maximum capacity of the queue used to prefetch\n                             assembled batches.\n    data_augmentation_options: a list of tuples, where each tuple contains a\n      data augmentation function and a dictionary containing arguments and their\n      values (see preprocessor.py).\n\n  Returns:\n    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts\n      (which hold images, boxes and targets).  To get a batch of tensor_dicts,\n      call input_queue.Dequeue().\n  '
    tensor_dict = create_tensor_dict_fn()
    tensor_dict[fields.InputDataFields.image] = tf.expand_dims(tensor_dict[fields.InputDataFields.image], 0)
    images = tensor_dict[fields.InputDataFields.image]
    float_images = tf.to_float(images)
    tensor_dict[fields.InputDataFields.image] = float_images
    include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks in tensor_dict)
    include_keypoints = (fields.InputDataFields.groundtruth_keypoints in tensor_dict)
    include_multiclass_scores = (fields.InputDataFields.multiclass_scores in tensor_dict)
    if data_augmentation_options:
        tensor_dict = preprocessor.preprocess(tensor_dict, data_augmentation_options, func_arg_map=preprocessor.get_default_func_arg_map(include_label_weights=True, include_multiclass_scores=include_multiclass_scores, include_instance_masks=include_instance_masks, include_keypoints=include_keypoints))
    input_queue = batcher.BatchQueue(tensor_dict, batch_size=batch_size_per_clone, batch_queue_capacity=batch_queue_capacity, num_batch_queue_threads=num_batch_queue_threads, prefetch_queue_capacity=prefetch_queue_capacity)
    return input_queue

def get_inputs(input_queue, num_classes, merge_multiple_label_boxes=False, use_multiclass_scores=False):
    'Dequeues batch and constructs inputs to object detection model.\n\n  Args:\n    input_queue: BatchQueue object holding enqueued tensor_dicts.\n    num_classes: Number of classes.\n    merge_multiple_label_boxes: Whether to merge boxes with multiple labels\n      or not. Defaults to false. Merged boxes are represented with a single\n      box and a k-hot encoding of the multiple labels associated with the\n      boxes.\n    use_multiclass_scores: Whether to use multiclass scores instead of\n      groundtruth_classes.\n\n  Returns:\n    images: a list of 3-D float tensor of images.\n    image_keys: a list of string keys for the images.\n    locations_list: a list of tensors of shape [num_boxes, 4]\n      containing the corners of the groundtruth boxes.\n    classes_list: a list of padded one-hot (or K-hot) float32 tensors containing\n      target classes.\n    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,\n      image_width] containing instance masks for objects if present in the\n      input_queue. Else returns None.\n    keypoints_list: a list of 3-D float tensors of shape [num_boxes,\n      num_keypoints, 2] containing keypoints for objects if present in the\n      input queue. Else returns None.\n    weights_lists: a list of 1-D float32 tensors of shape [num_boxes]\n      containing groundtruth weight for each box.\n  '
    read_data_list = input_queue.dequeue()
    label_id_offset = 1

    def extract_images_and_targets(read_data):
        'Extract images and targets from the input dict.'
        image = read_data[fields.InputDataFields.image]
        key = ''
        if (fields.InputDataFields.source_id in read_data):
            key = read_data[fields.InputDataFields.source_id]
        location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
        classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes], tf.int32)
        classes_gt -= label_id_offset
        if (merge_multiple_label_boxes and use_multiclass_scores):
            raise ValueError('Using both merge_multiple_label_boxes and use_multiclass_scores isnot supported')
        if merge_multiple_label_boxes:
            (location_gt, classes_gt, _) = util_ops.merge_boxes_with_multiple_labels(location_gt, classes_gt, num_classes)
            classes_gt = tf.cast(classes_gt, tf.float32)
        elif use_multiclass_scores:
            classes_gt = tf.cast(read_data[fields.InputDataFields.multiclass_scores], tf.float32)
        else:
            classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt, depth=num_classes, left_pad=0)
        masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
        keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
        if (merge_multiple_label_boxes and ((masks_gt is not None) or (keypoints_gt is not None))):
            raise NotImplementedError('Multi-label support is only for boxes.')
        weights_gt = read_data.get(fields.InputDataFields.groundtruth_weights)
        return (image, key, location_gt, classes_gt, masks_gt, keypoints_gt, weights_gt)
    return zip(*map(extract_images_and_targets, read_data_list))

def _create_losses(input_queue, create_model_fn, train_config):
    'Creates loss function for a DetectionModel.\n\n  Args:\n    input_queue: BatchQueue object holding enqueued tensor_dicts.\n    create_model_fn: A function to create the DetectionModel.\n    train_config: a train_pb2.TrainConfig protobuf.\n  '
    detection_model = create_model_fn()
    (images, _, groundtruth_boxes_list, groundtruth_classes_list, groundtruth_masks_list, groundtruth_keypoints_list, groundtruth_weights_list) = get_inputs(input_queue, detection_model.num_classes, train_config.merge_multiple_label_boxes, train_config.use_multiclass_scores)
    preprocessed_images = []
    true_image_shapes = []
    for image in images:
        (resized_image, true_image_shape) = detection_model.preprocess(image)
        preprocessed_images.append(resized_image)
        true_image_shapes.append(true_image_shape)
    images = tf.concat(preprocessed_images, 0)
    true_image_shapes = tf.concat(true_image_shapes, 0)
    if any(((mask is None) for mask in groundtruth_masks_list)):
        groundtruth_masks_list = None
    if any(((keypoints is None) for keypoints in groundtruth_keypoints_list)):
        groundtruth_keypoints_list = None
    detection_model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list, groundtruth_masks_list, groundtruth_keypoints_list, groundtruth_weights_list=groundtruth_weights_list)
    prediction_dict = detection_model.predict(images, true_image_shapes)
    losses_dict = detection_model.loss(prediction_dict, true_image_shapes)
    for loss_tensor in losses_dict.values():
        tf.losses.add_loss(loss_tensor)

def train(create_tensor_dict_fn, create_model_fn, train_config, master, task, num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name, is_chief, train_dir, graph_hook_fn=None):
    'Training function for detection models.\n\n  Args:\n    create_tensor_dict_fn: a function to create a tensor input dictionary.\n    create_model_fn: a function that creates a DetectionModel and generates\n                     losses.\n    train_config: a train_pb2.TrainConfig protobuf.\n    master: BNS name of the TensorFlow master to use.\n    task: The task id of this training instance.\n    num_clones: The number of clones to run per machine.\n    worker_replicas: The number of work replicas to train with.\n    clone_on_cpu: True if clones should be forced to run on CPU.\n    ps_tasks: Number of parameter server tasks.\n    worker_job_name: Name of the worker job.\n    is_chief: Whether this replica is the chief replica.\n    train_dir: Directory to write checkpoints and training summaries to.\n    graph_hook_fn: Optional function that is called after the inference graph is\n      built (before optimization). This is helpful to perform additional changes\n      to the training graph such as adding FakeQuant ops. The function should\n      modify the default graph.\n\n  Raises:\n    ValueError: If both num_clones > 1 and train_config.sync_replicas is true.\n  '
    detection_model = create_model_fn()
    data_augmentation_options = [preprocessor_builder.build(step) for step in train_config.data_augmentation_options]
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(num_clones=num_clones, clone_on_cpu=clone_on_cpu, replica_id=task, num_replicas=worker_replicas, num_ps_tasks=ps_tasks, worker_job_name=worker_job_name)
        with tf.device('/cpu:0'):
            global_step = slim.create_global_step()
        if ((num_clones != 1) and train_config.sync_replicas):
            raise ValueError('In Synchronous SGD mode num_clones must ', 'be 1. Found num_clones: {}'.format(num_clones))
        batch_size = (train_config.batch_size // num_clones)
        if train_config.sync_replicas:
            batch_size //= train_config.replicas_to_aggregate
        with tf.device('/cpu:0'):
            input_queue = create_input_queue(batch_size, create_tensor_dict_fn, train_config.batch_queue_capacity, train_config.num_batch_queue_threads, train_config.prefetch_queue_capacity, data_augmentation_options)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_summaries = set([])
        model_fn = functools.partial(_create_losses, create_model_fn=create_model_fn, train_config=train_config)
        clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
        first_clone_scope = clones[0].scope
        if graph_hook_fn:
            with tf.device('/cpu:0'):
                graph_hook_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        with tf.device('/cpu:0'):
            (training_optimizer, optimizer_summary_vars) = optimizer_builder.build(train_config.optimizer)
            for var in optimizer_summary_vars:
                tf.summary.scalar(var.op.name, var, family='LearningRate')
        sync_optimizer = None
        if train_config.sync_replicas:
            training_optimizer = npu_tf_optimizer(tf.train.SyncReplicasOptimizer(training_optimizer, replicas_to_aggregate=train_config.replicas_to_aggregate, total_num_replicas=worker_replicas))
            sync_optimizer = training_optimizer
        with tf.device('/cpu:0'):
            regularization_losses = (None if train_config.add_regularization_loss else [])
            (total_loss, grads_and_vars) = model_deploy.optimize_clones(clones, training_optimizer, regularization_losses=regularization_losses)
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')
            if train_config.bias_grad_multiplier:
                biases_regex_list = ['.*/biases']
                grads_and_vars = variables_helper.multiply_gradients_matching_regex(grads_and_vars, biases_regex_list, multiplier=train_config.bias_grad_multiplier)
            if train_config.freeze_variables:
                grads_and_vars = variables_helper.freeze_gradients_matching_regex(grads_and_vars, train_config.freeze_variables)
            if (train_config.gradient_clipping_by_norm > 0):
                with tf.name_scope('clip_grads'):
                    grads_and_vars = slim.learning.clip_gradient_norms(grads_and_vars, train_config.gradient_clipping_by_norm)
            grad_updates = training_optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops, name='update_barrier')
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')
        for model_var in slim.get_model_variables():
            global_summaries.add(tf.summary.histogram(('ModelVars/' + model_var.op.name), model_var))
        for loss_tensor in tf.losses.get_losses():
            global_summaries.add(tf.summary.scalar(('Losses/' + loss_tensor.op.name), loss_tensor))
        global_summaries.add(tf.summary.scalar('Losses/TotalLoss', tf.losses.get_total_loss()))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        summaries |= global_summaries
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        if True:
            session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
        keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        init_fn = None
        if train_config.fine_tune_checkpoint:
            if (not train_config.fine_tune_checkpoint_type):
                if train_config.from_detection_checkpoint:
                    train_config.fine_tune_checkpoint_type = 'detection'
                else:
                    train_config.fine_tune_checkpoint_type = 'classification'
            var_map = detection_model.restore_map(fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type, load_all_detection_checkpoint_vars=train_config.load_all_detection_checkpoint_vars)
            available_var_map = variables_helper.get_variables_available_in_checkpoint(var_map, train_config.fine_tune_checkpoint, include_global_step=False)
            init_saver = tf.train.Saver(available_var_map)

            def initializer_fn(sess):
                init_saver.restore(sess, train_config.fine_tune_checkpoint)
            init_fn = initializer_fn
        slim.learning.train(train_tensor, logdir=train_dir, master=master, is_chief=is_chief, session_config=session_config, startup_delay_steps=train_config.startup_delay_steps, init_fn=init_fn, summary_op=summary_op, number_of_steps=(train_config.num_steps if train_config.num_steps else None), save_summaries_secs=120, sync_optimizer=sync_optimizer, saver=saver)
