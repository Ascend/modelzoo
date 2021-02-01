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
import time
import logging
import warnings
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.compat import compat

from config.db_config import cfg
from data.generator import generator as generate
from networks.model import dbnet as DBNet
from networks.losses import db_loss, db_acc
from networks.learning_rate import learning_rate_with_decay, learning_rate_with_exponential_decay

warnings.filterwarnings("ignore")

import npu_bridge 
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def train_logger_init():
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.DEBUG)

    log_file = os.path.join(cfg.TRAIN.TRAIN_LOGS, time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    train_logger.addHandler(file_handler)

    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.DEBUG)
    consol_formatter = logging.Formatter('%(message)s')
    consol_handler.setFormatter(consol_formatter)
    train_logger.addHandler(consol_handler)
    return train_logger


def tower_loss(images, gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask, gt_topk_masks, reuse_variables):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        with compat.forward_compatibility_horizon(2019, 5, 1):
            binarize_map, threshold_map, thresh_binary = DBNet(images)

    model_loss = db_loss(binarize_map, threshold_map, thresh_binary, gt_score_maps,
                         gt_threshold_map, gt_score_mask, gt_thresh_mask, gt_topk_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
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


def main():
    make_dir(cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR)
    make_dir(cfg.TRAIN.TRAIN_LOGS)

    train_logger = train_logger_init()
    b_s = cfg.TRAIN.BATCH_SIZE
    i_s = cfg.TRAIN.IMG_SIZE

    input_images = tf.placeholder(tf.float32, shape=[b_s, i_s, i_s, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[b_s, i_s, i_s, 1], name='input_score_maps')
    input_threshold_maps = tf.placeholder(tf.float32, shape=[b_s, i_s, i_s, 1], name='input_threshold_maps')
    input_score_masks = tf.placeholder(tf.float32, shape=[b_s, i_s, i_s, 1], name='input_score_masks')
    input_threshold_masks = tf.placeholder(tf.float32, shape=[b_s, i_s, i_s, 1], name='input_threshold_masks')
    input_topk_masks = tf.placeholder(tf.float32, shape=[b_s * i_s * i_s], name='input_topk_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    if cfg.LR == 'exponential_decay':
        learning_rate_fn = learning_rate_with_exponential_decay()
    elif cfg.LR == 'paper_decay':
        learning_rate_fn = learning_rate_with_decay(start_lr=0.0035, power=0.9)
    else:
        assert 0, 'error Learning_rate'

    learning_rate = learning_rate_fn(global_step)
    tf.summary.scalar('learning rate', learning_rate)

    if cfg.TRAIN.OPT == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif cfg.TRAIN.OPT == 'sgd':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    else:
        assert 0, 'error_optimzer'

    reuse_variables = None

    with tf.name_scope('model') as scope:

        total_loss, model_loss, binarize_map, threshold_map, thresh_binary = \
            tower_loss(input_images, input_score_maps, input_threshold_maps, input_score_masks,
                       input_threshold_masks, input_topk_masks, reuse_variables)

        binarize_acc, thresh_binary_acc = db_acc(binarize_map, threshold_map, thresh_binary,
                                                 input_score_maps, input_threshold_maps, input_score_masks,
                                                 input_threshold_masks)

        reuse_variables = True
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))

        if cfg.PLATFORM == "GPU":
            grads = opt.compute_gradients(total_loss)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        elif cfg.PLATFORM == "NPU":
            # grads = opt.compute_gradients(total_loss)
            # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            loss_scaling = 2 ** 12
            grads = opt.compute_gradients(total_loss * loss_scaling)
            grads = [(grad / loss_scaling, var) for grad, var in grads]
            grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads if grad is not None]
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        else:
            assert 0, 'Wrong Platform!'

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(cfg.TRAIN.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.TRAIN.SAVE_MAX)
    train_logs_dir = os.path.join(cfg.TRAIN.TRAIN_LOGS, 'train')
    make_dir(train_logs_dir)
    train_summary_writer = tf.summary.FileWriter(train_logs_dir, tf.get_default_graph())
    init = tf.global_variables_initializer()
    
    if cfg.PLATFORM == "GPU":
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif cfg.PLATFORM == "NPU":
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        pass

    with tf.Session(config=config) as sess:
        try:
            if cfg.TRAIN.RESTORE:
                train_logger.info('continue training from previous checkpoint')
                ckpt = tf.train.get_checkpoint_state(cfg.TRAIN.RESTORE_CKPT_PATH)
                saver.restore(sess, ckpt.model_checkpoint_path)
                train_logger.info("done")
            elif cfg.TRAIN.PRETRAINED_MODEL_PATH is not None:
                sess.run(init)
                train_logger.info('loading pretrain model')
                variable_restore_op = slim.assign_from_checkpoint_fn(cfg.TRAIN.PRETRAINED_MODEL_PATH,
                                                                     slim.get_trainable_variables(),
                                                                     ignore_missing_vars=True)
                variable_restore_op(sess)
                train_logger.info("done")
            else:
                sess.run(init)
        except ValueError:
            assert 0, 'load error'

        train_data_generator = generate()

        start = time.time()
        for step in range(cfg.TRAIN.MAX_STEPS):
            train_data = next(train_data_generator)

            train_feed_dict = {input_images: train_data[0],
                               input_score_maps: (train_data[1][..., np.newaxis]).copy(),
                               input_score_masks: (train_data[2][..., np.newaxis]).copy(),
                               input_threshold_maps: (train_data[3][..., np.newaxis]).copy(),
                               input_threshold_masks: (train_data[4][..., np.newaxis]).copy(),
                               input_topk_masks: train_data[5]}

            if step == 0:
                train_logger.info('start training')
                continue

            if step % cfg.TRAIN.SAVE_SUMMARY_STEPS == 0:
                ml, tl, _, train_summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                                        feed_dict=train_feed_dict)
                train_summary_writer.add_summary(train_summary_str, global_step=step)
            else:
                ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict=train_feed_dict)

            if np.isnan(tl):
                train_logger.info('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * cfg.TRAIN.BATCH_SIZE) / (time.time() - start)
                start = time.time()
                train_logger.info(
                    '{}->Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        cfg.TRAIN.OPT, step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % cfg.TRAIN.SAVE_CHECKPOINT_STEPS == 0:
                saver.save(sess, os.path.join(cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR,
                                              'DB_' + cfg.BACKBONE + '_' + cfg.TRAIN.OPT + '_model.ckpt'),
                           global_step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change db_config file')

    parser.add_argument('--max_steps', '-m', default=188250, type=int, help='max_steps integer')
    parser.add_argument('--save_steps', '-s', default=3000, type=int, help='save_steps integer')
    parser.add_argument('--learning_rate', '-lr', default=0.0035, type=float, help='learning rate')
    parser.add_argument('--platform', '-p', default="NPU", type=str, help='NPU or GPU')

    args = parser.parse_args()
    cfg.TRAIN.MAX_STEPS = args.max_steps
    cfg.TRAIN.SAVE_CHECKPOINT_STEPS = args.save_steps
    cfg.PLATFORM = args.platform
    cfg.TRAIN.LEARNING_RATE = args.learning_rate

    main()
