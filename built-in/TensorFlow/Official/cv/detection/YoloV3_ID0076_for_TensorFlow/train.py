# coding=utf-8
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

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange
import random
import time

from utils.data_utils import get_batch_data, color_jitter
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec

from model import yolov3
import time
import os

# npu modified
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu import util

import argparse

parser = argparse.ArgumentParser(description="YOLO-V3 training setting.")
parser.add_argument("--mode", type=str, default='single',
                    help="setting train mode of training.")
parser.add_argument("--resume", type=bool, default=False,
                    help="setting if train from resume.")
parser.add_argument("--data_url", default='/cache/data_url',
                    help="setting dir of training data.")
parser.add_argument("--train_url", default='/cache/training',
                    help="setting dir of training output.")
parser.add_argument("--train_file", default='',
                    help="path of train annotation file.")
parser.add_argument("--save_dir", default='./training/',
                    help="path of ckpt.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="batchsize.")

# modify for npu overflow start
# enable overflow
parser.add_argument("--over_dump", type=str, default="False",
                    help="whether to enable overflow")
parser.add_argument("--over_dump_path", type=str, default="./",
                    help="path to save overflow dump files")
# modify for npu overflow end

args_input = parser.parse_args()

if args_input.mode == 'single':
    import args_single as args
elif args_input.mode == 'multi':
    import args_multi as args
elif args_input.mode == 'modelarts_single':
    import moxing as mox
    import modelarts.frozen_graph as fg
    import modelarts.args_modelarts_single as args
elif args_input.mode == 'modelarts_multi':
    import moxing as mox
    import modelarts.frozen_graph as fg
    import modelarts.args_modelarts_multi as args

if args_input.mode == 'modelarts_single' or args_input.mode == 'modelarts_multi':
    real_path = '/cache/data_url'
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(args_input.data_url, real_path)
    print('training data finish copy to %s.' % real_path)

if args_input.train_file:
    args.train_file = args_input.train_file
if args_input.save_dir:
    args.save_dir = args_input.save_dir
if args_input.batch_size:
    args.batch_size = args_input.batch_size

print('setting train mode %s.' % args_input.mode)

# setting loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')

##################
# tf.data pipeline
##################
train_dataset = tf.data.TextLineDataset(args.train_file)
print('##########################args_input.rank_id', os.environ['RANK_ID'])
logging.info('shuffle seed_%s args.', os.environ['RANK_ID'])

train_dataset = train_dataset.shuffle(args.train_img_cnt, seed=int(os.environ['DEVICE_ID']),
                                      reshuffle_each_iteration=True)
print('##########################args.train_img_cnt', args.train_img_cnt)

train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)  # npu modified
train_dataset = train_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train,
                              args.use_mix_up, args.letterbox_resize],
                         Tout=[tf.float32,
                               tf.float32, tf.float32, tf.float32,
                               tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=20
)


def valid_shape(*x):
    image, y_true_13, y_true_26, y_true_52, gt_box_13, gt_box_26, gt_box_52 = x
    y_true = [y_true_13, y_true_26, y_true_52]
    gt_box = [gt_box_13, gt_box_26, gt_box_52]

    # npu modified
    if args_input.mode == 'single' or args_input.mode == 'modelarts_single':
        image.set_shape([args.batch_size, args.img_size[0], args.img_size[1], 3])
        y_true[0].set_shape([args.batch_size, 13, 13, 3, args.class_num + 5 + 1])
        y_true[1].set_shape([args.batch_size, 26, 26, 3, args.class_num + 5 + 1])
        y_true[2].set_shape([args.batch_size, 52, 52, 3, args.class_num + 5 + 1])
    elif args_input.mode == 'multi' or args_input.mode == 'modelarts_multi':
        image.set_shape([args.batch_size, args.img_size[0], args.img_size[1], 3])
        y_true[0].set_shape([args.batch_size, 19 * 1, 19 * 1, 3, args.class_num + 5 + 1])
        y_true[1].set_shape([args.batch_size, 19 * 2, 19 * 2, 3, args.class_num + 5 + 1])
        y_true[2].set_shape([args.batch_size, 19 * 4, 19 * 4, 3, args.class_num + 5 + 1])

    gt_box[0].set_shape([args.batch_size, 1, 32, 4])
    gt_box[1].set_shape([args.batch_size, 1, 64, 4])
    gt_box[2].set_shape([args.batch_size, 1, 128, 4])

    image = color_jitter(
        image, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)

    return image, y_true_13, y_true_26, y_true_52, gt_box_13, gt_box_26, gt_box_52


train_dataset = train_dataset.map(valid_shape, num_parallel_calls=20)
train_dataset = train_dataset.prefetch(args.prefetech_buffer)
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init_op = iterator.make_initializer(train_dataset)
# get an element from the chosen dataset iterator
image, y_true_13, y_true_26, y_true_52, gt_box_13, gt_box_26, gt_box_52 = iterator.get_next()
y_true = [y_true_13, y_true_26, y_true_52]
gt_box = [gt_box_13, gt_box_26, gt_box_52]

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay,
                    args.weight_decay, use_static_shape=False,
                    batch_size=args.batch_size, img_size=args.img_size)

with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=True)
loss = yolo_model.compute_loss(pred_feature_maps, y_true, gt_box)
l2_loss = tf.losses.get_regularization_loss()

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(
    var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)

tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])


def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 0.08 epoch warm up."""
    initial_learning_rate = args.learning_rate_init
    batches_per_epoch = args.train_batch_num // args.iterations_per_loop * args.iterations_per_loop
    total_steps = int(args.total_epoches * batches_per_epoch)
    warmup_steps = int(batches_per_epoch * args.warm_up_epoch)
    tf.compat.v1.logging.info('total_steps: %d', int(total_steps))
    tf.compat.v1.logging.info('warmup_steps: %d', int(warmup_steps))
    lr = tf.maximum(
        tf.compat.v1.train.cosine_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step - warmup_steps,
            decay_steps=total_steps - warmup_steps,
        ),
        0,
    )
    warmup_lr = (
            initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
        warmup_steps, tf.float32))
    return tf.cond(pred=global_step < warmup_steps,
                   true_fn=lambda: warmup_lr,
                   false_fn=lambda: lr)


global_step = tf.train.get_or_create_global_step()
learning_rate = learning_rate_fn(global_step)
tf.summary.scalar('learning_rate', learning_rate)

if not args.save_optimizer:
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

optimizer = config_optimizer(args.optimizer_name, learning_rate)
optimizer = NPUDistributedOptimizer(optimizer)
loss_scale_manager = FixedLossScaleManager(loss_scale=128)
if args.num_gpus > 1:
    optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=True)
else:
    optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=False)

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # apply gradient clip to avoid gradient exploding
    gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
    clip_grad_var = [gv if gv[0] is None else [
        tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
    train_op = optimizer.apply_gradients(clip_grad_var, global_step=tf.train.get_global_step())

if args.save_optimizer:
    print(
        'Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

# npu modified
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # training on Ascend chips
custom_op.parameter_map["enable_data_pre_proc"].b = True
custom_op.parameter_map["iterations_per_loop"].i = args.iterations_per_loop

#  init autotune module start
autotune = False
autotune = os.environ.get('autotune')
if autotune:
    autotune = autotune.lower()
    if autotune == 'true':
        print("Autotune module is :" + autotune)
        print("Autotune module has been initiated!")
        custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
    else:
        print("Autotune module is :" + autotune)
        print("Autotune module is enabled or with error setting.")
else:
    print("Autotune module de_initiate!Pass")
#  init autotune module end

if args_input.over_dump == "True":
    print("NPU overflow dump is enabled")
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args_input.over_dump_path)
    custom_op.parameter_map["enable_dump_debug"].b = True
    custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
else:
    print("NPU overflow dump is disabled")

config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    # yolov3 finetuning璁粌寮鍚紙darknet53.ckpt锛
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    # 鏂偣缁寮鍚
    if args_input.resume:
        saver_to_restore = tf.train.Saver()
        saver_to_restore.restore(sess, tf.train.latest_checkpoint(args.save_dir))
    else:
        saver_to_restore.restore(sess, args.restore_path)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    print('\n----------- start to train -----------\n')

    best_mAP = -np.Inf
    train_op = util.set_iteration_per_loop(sess, train_op, args.iterations_per_loop)
    sess.run(train_init_op)
    for epoch in range(args.total_epoches):
        loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for i in trange(args.train_batch_num // args.iterations_per_loop):
            t = time.time()
            _, summary, __y_true, __loss, __global_step, __lr = sess.run(
                [train_op, merged, y_true, loss, global_step, learning_rate]
            )
            fps = 1 / (time.time() - t) * args.iterations_per_loop * args.num_gpus * args.batch_size

            writer.add_summary(summary, global_step=__global_step)

            loss_total.update(__loss[0], len(__y_true[0]))
            loss_xy.update(__loss[1], len(__y_true[0]))
            loss_wh.update(__loss[2], len(__y_true[0]))
            loss_conf.update(__loss[3], len(__y_true[0]))
            loss_class.update(__loss[4], len(__y_true[0]))

            info = "Epoch: {}, global_step: {} fps: {:.2f} lr: {:.6f} | loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(
                epoch, int(__global_step), fps, __lr, loss_total.average, loss_xy.average, loss_wh.average,
                loss_conf.average,
                loss_class.average)
            print(info)
            logging.info(info)

        # NOTE: this is just demo. You can set the conditions when to save the weights.
        temp_epoch = epoch + 1
        if temp_epoch % args.save_epoch == 0 and epoch > 0:
            saver_to_save.save(sess, args.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format( \
                temp_epoch,
                int(__global_step),
                loss_total.average,
                __lr))

        if __lr <= 0:
            break

    saver_to_save.save(sess, args.save_dir + 'model-final_step_{}_loss_{:.4f}_lr_{:.5g}'.format( \
        int(__global_step),
        loss_total.average,
        __lr))

if args_input.mode == 'modelarts_single' or args_input.mode == 'modelarts_multi':
    model_args = {}
    model_args['ckpt_path'] = os.path.join(args.save_dir, 'model-final_step_{}_loss_{:.4f}_lr_{:.5g}'.format(
        int(__global_step), loss_total.average, __lr))
    model_args['anchor_path'] = args.anchor_path
    model_args['class_name_path'] = args.class_name_path
    fg.freeze_graph_def(**model_args)
    # model copy to train_url
    mox.file.copy_parallel('/cache/training/', args_input.train_url)
