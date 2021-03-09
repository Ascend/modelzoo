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
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
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
import argparse

import numpy as np
import random

np.random.seed(5)  # seed是一个固定的整数即可
random.seed(5)
os.environ['PYTHONHASHSEED'] = str(5)
import time

import tensorflow as tf
from tensorflow.python.compat import compat

from easydict import EasyDict as edict
from model.custom import siammask
from preprocessing.generator import build_data_loader
from model.lossy import total_loss
from utils.config_helper import load_config


def parser_():
    parser = argparse.ArgumentParser(description='Tensorflow Tracking SiamMask Training')
    parser.add_argument('--config', dest='config', default="./config/config.json",
                        help='hyperparameter of SiamMask in json format')
    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--log', default="./logs/", type=str,
                        help='log file')
    parser.add_argument('--TRAIN_MAX_STEPS', default=750000, type=int, metavar='N',
                        help='number of total steps to run')
    parser.add_argument('--SAVE_MAX', default=1000, type=int, metavar='N',
                        help='number of total ckpt to save')
    parser.add_argument('--model_checkpoint_path', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--platform', default="NPU", type=str, help='NPU or GPU')
    parser.add_argument('--optimizer', default="sgd", type=str, help='sgd or adam')

    return parser.parse_args()


args = parser_()
args = vars(args)
args = edict(args)

if args["platform"] == "NPU":
    import npu_bridge
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def train(train_loader, cfg):
    template_size = 127
    search_size = 255
    cls_size = 25
    batch_size = args["batch_size"]
    print("batch_size", batch_size)
    log_path = args.log

    make_dir(log_path)
    train_logs_dir = os.path.join(log_path, 'train')
    make_dir(train_logs_dir)
    checkpoints_output_dir = os.path.join(log_path, 'ckpt')
    make_dir(checkpoints_output_dir)

    template = tf.placeholder(tf.float32, shape=[batch_size, template_size, template_size, 3], name='template')
    search = tf.placeholder(tf.float32, shape=[batch_size, search_size, search_size, 3], name='search')
    label_cls = tf.placeholder(tf.float32, shape=[batch_size, cls_size, cls_size, 5], name='label_cls')
    label_loc = tf.placeholder(tf.float32, shape=[batch_size, cls_size, cls_size, 4, 5], name='label_loc')
    label_loc_weight = tf.placeholder(tf.float32, shape=[batch_size, cls_size, cls_size, 5], name='label_loc_weight')
    label_mask = tf.placeholder(tf.float32, shape=[batch_size, search_size, search_size, 1], name='label_mask')
    label_mask_weight = tf.placeholder(tf.float32, shape=[batch_size, cls_size, cls_size, 1], name='label_mask_weight')

    from utils.lr_helper import learning_rate_with_decay_16 as learning_rate_with_decay
    learning_rate_fn = learning_rate_with_decay()

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    with tf.name_scope('model') as scope:
        with compat.forward_compatibility_horizon(2019, 5, 1):
            rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = siammask(template, search,
                                                                                                   True)
        rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = total_loss(
            label_cls,
            label_loc,
            label_loc_weight,
            label_mask,
            label_mask_weight,
            rpn_pred_cls,
            rpn_pred_loc,
            rpn_pred_mask)

        rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = tf.reduce_mean(rpn_loss_cls), tf.reduce_mean(
            rpn_loss_loc), tf.reduce_mean(rpn_loss_mask)

        mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = tf.reduce_mean(iou_acc_mean), tf.reduce_mean(
            iou_acc_5), tf.reduce_mean(iou_acc_7)

        cls_weight, reg_weight, mask_weight = cfg['loss']['weight']
        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_mask_loss * mask_weight

        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))

        learning_rate = learning_rate_fn(global_step)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=args.SAVE_MAX)

        if args["optimizer"] == "sgd":
            opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        elif args["optimizer"] == "adam":
            opt = tf.train.AdamOptimizer(learning_rate)
        else:
            assert 0, 'There is no corresponding optimizer'
        grads = opt.compute_gradients(loss)

        tf.summary.scalar('losses/learning_rate', learning_rate)
        tf.summary.scalar('losses/loss_loss', loss)
        tf.summary.scalar('losses/loss_rpn_cls_loss', rpn_cls_loss)
        tf.summary.scalar('losses/loss_rpn_loc_loss', rpn_loc_loss)
        tf.summary.scalar('losses/loss_rpn_mask_loss', rpn_mask_loss)
        tf.summary.scalar('losses/mask_iou_mean', mask_iou_mean)
        tf.summary.scalar('losses/mask_iou_at_5', mask_iou_at_5)
        tf.summary.scalar('losses/mask_iou_at_7', mask_iou_at_7)
        merged = tf.summary.merge_all()

    apply_gradient_op = opt.apply_gradients(grads, global_step)
    variables_averages_op = tf.group(*tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    init = tf.global_variables_initializer()

    if args["platform"] == "GPU":
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif args["platform"] == "NPU":
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
    else:
        assert 0, 'There is no corresponding platform'

    with tf.Session(config=config) as sess:
        train_data_generator = train_loader
        if args.model_checkpoint_path is not None:
            saver.restore(sess, args.model_checkpoint_path)
        else:
            sess.run(init)
        write = tf.summary.FileWriter(train_logs_dir, sess.graph)
        for step in range(args.TRAIN_MAX_STEPS):
            start = time.time()
            template_batch, search_batch, label_cls_batch, label_loc_batch, label_loc_weight_batch, \
            label_mask_batch, label_mask_weight_batch = next(train_data_generator)
            train_feed_dict = {template: template_batch,
                               search: search_batch,
                               label_cls: label_cls_batch,
                               label_loc: label_loc_batch,
                               label_loc_weight: label_loc_weight_batch,
                               label_mask: label_mask_batch,
                               label_mask_weight: label_mask_weight_batch
                               }
            summery, loss_val, rpn_cls_loss_val, rpn_loc_loss_val, rpn_mask_loss_val, train_op_val = sess.run(
                [merged, loss, rpn_cls_loss, rpn_loc_loss, rpn_mask_loss, train_op], feed_dict=train_feed_dict)
            end = time.time()
            print(
                "step : {}>>>>loss_val : {}， rpn_cls_loss : {}， rpn_loc_loss : {}，rpn_mask_loss : {}，cost_time : {}".format(
                    step,
                    loss_val,
                    rpn_cls_loss_val,
                    rpn_loc_loss_val,
                    rpn_mask_loss_val,
                    end - start,
                ))
            write.add_summary(summery, step)
            if step % 1000 == 0:
                saver.save(sess, os.path.join(checkpoints_output_dir,
                                              'siamMask_model.ckpt'), global_step=step)
    write.close()


def main():
    cfg = load_config(args)
    train_loader = build_data_loader(cfg, epochs=args["epochs"], batch_size=args["batch_size"])
    train(train_loader, cfg)


if __name__ == '__main__':
    main()
