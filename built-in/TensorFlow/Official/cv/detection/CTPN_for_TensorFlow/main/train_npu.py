#
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
#

import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np
sys.path.append(os.getcwd())

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)

from tensorflow.contrib import slim

tf.app.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.app.flags.DEFINE_integer('max_steps', 50000, '')
tf.app.flags.DEFINE_integer('decay_steps', 30000, '')
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_readers', 4, '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path',"checkpoints_mlt/" , '')
tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')
tf.app.flags.DEFINE_boolean('restore', False, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
tf.app.flags.DEFINE_string('dataset_dir', 'resized/', '')
tf.app.flags.DEFINE_integer('num_bbox', 256, '')
tf.app.flags.DEFINE_integer('loss_scale', 4096, '')
tf.app.flags.DEFINE_integer('inputs_height', 600, '')
tf.app.flags.DEFINE_integer('inputs_width', 900, '')

# modify for npu overflow start
# enable overflow
tf.app.flags.DEFINE_string("over_dump", "False",
                           "whether to enable overflow")
tf.app.flags.DEFINE_string("over_dump_path", "./",
                    "path to save overflow dump files")
# modify for npu overflow end

FLAGS = tf.app.flags.FLAGS


from nets import model_train as model
from utils.dataset import data_provider as data_provider

# npu libs
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import allreduce
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager


from tensorflow.python.client import timeline



def pad_input(inputs,target_shape=[1216,1216,3]):

    h,w = inputs.shape[:2]
    out = np.zeros(target_shape).astype(np.uint8)
    out[0:h,0:w,:] = inputs

    return out


def pad_bbox(inputs, count=256):
    if len(inputs)>count:
        return inputs[:count].copy()
   
    else:    
        out = inputs.copy()
        num_inputs = len(out)
        num_pad = count - num_inputs
        
        for i in range(num_pad):
            out.append([0,0,0,0,1])
        return out




def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.placeholder(tf.float32, 
            shape=[1,FLAGS.inputs_height, FLAGS.inputs_width, 3], 
            name='input_image')
    input_bbox = tf.placeholder(tf.float32, 
            shape=[FLAGS.num_bbox, 5], name='input_bbox')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    opt = NPUDistributedOptimizer(opt)
    loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.loss_scale)
    opt = NPULossScaleOptimizer(opt, loss_scale_manager)
    

    with tf.name_scope('model' ) as scope:
        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss_v2(bbox_pred, cls_pred, input_bbox)
                                                                             
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    
    if FLAGS.over_dump == "True":
        print("NPU overflow dump is enabled")
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.over_dump_path)
        custom_op.parameter_map["enable_dump_debug"].b = True
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    else:
        print("NPU overflow dump is disabled")
    
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers)
        start = time.time()

        for step in range(restore_step, FLAGS.max_steps):
            data = next(data_generator)
            inputs_padded = data[0]
            bbox_padded = pad_bbox(data[1],FLAGS.num_bbox)
            input_image_np = inputs_padded
            input_bbox_np = bbox_padded
            
            ml, tl,ce_loss, bbox_loss, _, summary_str = sess.run([model_loss, total_loss,
                                               rpn_cross_entropy,
                                               rpn_loss_box,
                                               train_op, summary_op],
                                              feed_dict={input_image: input_image_np,
                                                         input_bbox: input_bbox_np})
            summary_writer.add_summary(summary_str, global_step=step)
            print('model loss :', ml, 'ce_loss: ', ce_loss, 'box_loss:',bbox_loss)
            if step != 0 and step % FLAGS.decay_steps == 0:
                sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {:06d}, ce_loss {:.6f}, bbox_loss {:.6f}  model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    step, ce_loss, bbox_loss, ml, tl, avg_time_per_step, learning_rate.eval()))

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))

if __name__ == '__main__':
    tf.app.run()
