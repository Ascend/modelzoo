# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import logging
import time
from config.db_config import cfg
import warnings
from data.generator import generator as generate

from networks.model import dbnet as DBNet
from networks.losses import db_loss, db_acc
from tensorflow.python.compat import compat
from tensorflow.python import debug as tf_debug
from cv2 import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

np.random.seed(1)
from tensorflow import set_random_seed

set_random_seed(2)


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def _train_logger_init():
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.DEBUG)

    # 添加文件输出
    log_file = os.path.join(cfg["TRAIN"]["TRAIN_LOGS"],
                            time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    train_logger.addHandler(file_handler)

    # 添加控制台输出
    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.DEBUG)
    consol_formatter = logging.Formatter('%(message)s')
    consol_handler.setFormatter(consol_formatter)
    train_logger.addHandler(consol_handler)
    return train_logger


def tower_loss(images, gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask, reuse_variables):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        with compat.forward_compatibility_horizon(2019, 5, 1):
            binarize_map, threshold_map, thresh_binary = DBNet(images)

    model_loss = db_loss(binarize_map, threshold_map, thresh_binary, gt_score_maps, \
                         gt_threshold_map, gt_score_mask, gt_thresh_mask)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    if reuse_variables is None:
        tf.summary.image('gt/input_imgs', images)
        tf.summary.image('gt/score_map', gt_score_maps)
        tf.summary.image('gt/threshold_map', gt_threshold_map * 255)
        tf.summary.image('gt/score_mask', gt_score_mask)
        tf.summary.image('gt/thresh_mask', gt_thresh_mask)

        tf.summary.image('pred/binarize_map', binarize_map)
        tf.summary.image('pred/threshold_map', threshold_map * 255)
        tf.summary.image('pred/thresh_binary', thresh_binary)

        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, binarize_map, threshold_map, thresh_binary


def parse_function(example_proto):
    dics = {
        'input_images': tf.FixedLenFeature(shape=(8, 640, 640, 3), dtype=tf.float32),
        'input_score_maps': tf.FixedLenFeature(shape=(8, 640, 640), dtype=tf.float32),
        'input_score_masks': tf.FixedLenFeature(shape=(8, 640, 640), dtype=tf.float32),
        'input_threshold_maps': tf.FixedLenFeature(shape=(8, 640, 640), dtype=tf.float32),
        'input_threshold_masks': tf.FixedLenFeature(shape=(8, 640, 640), dtype=tf.float32)
    }
    parsed_example = tf.parse_single_example(example_proto, dics)
    return parsed_example


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main():
    make_dir(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"])
    make_dir(cfg["TRAIN"]["TRAIN_LOGS"])

    train_logger = _train_logger_init()

    input_images = tf.placeholder(tf.float32, shape=[8, 640, 640, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[8, 640, 640, 1], name='input_score_maps')
    input_threshold_maps = tf.placeholder(tf.float32, shape=[8, 640, 640, 1], name='input_threshold_maps')
    input_score_masks = tf.placeholder(tf.float32, shape=[8, 640, 640, 1], name='input_score_masks')
    input_threshold_masks = tf.placeholder(tf.float32, shape=[8, 640, 640, 1], name='input_threshold_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(cfg["TRAIN"]["LEARNING_RATE"], global_step, decay_steps=10000,
                                               decay_rate=0.9, staircase=True)
    if cfg.TRAIN.OPT == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif cfg.TRAIN.OPT == 'SGD':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    else:
        assert 0, 'error_optimzer'

    tf.summary.scalar('learning rate', learning_rate)

    grads = None
    tower_grads = []
    reuse_variables = None

    with tf.name_scope('model') as scope:
        gt_imgs = input_images
        gt_scores = input_score_maps
        gt_thresholds = input_threshold_maps
        gt_score_masks = input_score_masks
        gt_threshold_masks = input_threshold_masks
        total_loss, model_loss, binarize_map, threshold_map, thresh_binary = \
            tower_loss(gt_imgs, gt_scores, gt_thresholds, gt_score_masks, gt_threshold_masks, reuse_variables)
        binarize_acc, thresh_binary_acc = db_acc(binarize_map, threshold_map, thresh_binary,
                                                 gt_scores, gt_thresholds, gt_score_masks,
                                                 gt_threshold_masks)

        reuse_variables = True
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        grads = opt.compute_gradients(total_loss)

    grad_output = grads

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()

    variable_averages = tf.train.ExponentialMovingAverage(cfg.TRAIN.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.TRAIN.SAVE_MAX)

    train_logs_dir = os.path.join(cfg.TRAIN.TRAIN_LOGS, 'train')
    val_logs_dir = os.path.join(cfg.TRAIN.TRAIN_LOGS, 'val')

    make_dir(train_logs_dir)
    make_dir(val_logs_dir)

    train_summary_writer = tf.summary.FileWriter(train_logs_dir, tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(val_logs_dir, tf.get_default_graph())

    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # config = tf.ConfigProto()
    # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name = "NpuOptimizer"
    # custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
    # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关

    with tf.Session(config=config) as sess:

        try:
            if cfg["TRAIN"]["RESTORE"]:
                train_logger.info('continue training from previous checkpoint')
                ckpt = tf.train.get_checkpoint_state(cfg["TRAIN"]["RESTORE_CKPT_PATH"])
                train_logger.info('restore model path:', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                train_logger.info("done")
            elif cfg["TRAIN"]["PRETRAINED_MODEL_PATH"] is not None:
                sess.run(init)
                train_logger.info('load pretrain model:{}', str(cfg["TRAIN"]["PRETRAINED_MODEL_PATH"]))
                variable_restore_op = slim.assign_from_checkpoint_fn(cfg["TRAIN"]["PRETRAINED_MODEL_PATH"],
                                                                     slim.get_trainable_variables(),
                                                                     ignore_missing_vars=True)
                variable_restore_op(sess)
                train_logger.info("done")
            else:
                sess.run(init)
        except:
            assert 0, 'load error'

        filenames = ["no_random_10.tfrecord"]
        dataset = tf.data.TFRecordDataset(filenames)
        new_dataset = dataset.map(parse_function)
        iterator = new_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        test_epoch = 0

        start = time.time()
        for step in range(cfg.TRAIN.MAX_STEPS):
            try:
                train_data = sess.run([next_element['input_images'], next_element['input_score_maps'],
                                       next_element['input_score_masks'], next_element['input_threshold_maps'],
                                       next_element['input_threshold_masks']])
                # cv2.imwrite("./temp/"+str(step)+".jpg", np.abs(train_data[0][0]*255))
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break

            train_feed_dict = {input_images: train_data[0],
                               input_score_maps: (train_data[1][..., np.newaxis]).copy(),
                               input_score_masks: (train_data[2][..., np.newaxis]).copy(),
                               input_threshold_maps: (train_data[3][..., np.newaxis]).copy(),
                               input_threshold_masks: (train_data[4][..., np.newaxis]).copy()}
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
            ml, tl, gr, _ = sess.run([model_loss, total_loss, grad_output, train_op], feed_dict=train_feed_dict)

            print_grad = []
            if step == 2:
                for g, _ in gr:
                    print_grad.append(g)
                m = np.array(print_grad)
                np.save("gpu_with_avg.npy", m)

            if np.isnan(tl):
                train_logger.info('Loss diverged, stop training')
                break
            if step % 2 == 0:
                avg_time_per_step = (time.time() - start) / 2
                avg_examples_per_second = (2 * cfg.TRAIN.BATCH_SIZE) / (time.time() - start)
                start = time.time()
                train_logger.info(
                    '{}->Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        cfg.TRAIN.OPT, step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % cfg.TRAIN.SAVE_CHECKPOINT_STEPS == 0:
                saver.save(sess, os.path.join(cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR,
                                              'DB_' + cfg.BACKBONE + '_' + cfg.TRAIN.OPT + '_model.ckpt'),
                           global_step=step)


if __name__ == '__main__':
    main()