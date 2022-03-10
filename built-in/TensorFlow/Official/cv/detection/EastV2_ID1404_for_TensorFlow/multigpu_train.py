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

from npu_bridge.npu_init import *
import os
import tensorflow as tf
from tensorflow.contrib import slim

import config as cfg
from timers import *
import model
import loss
import dataset
import queue
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_list
gpus = list(range(len(cfg.gpu_list.split(','))))

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("mul_rank_size", 1, "number of npu device")
flags.DEFINE_integer("mul_device_id", 0, "indicator of npu device")

image_list = np.array(dataset.get_datalist(cfg.train_data_path))
if FLAGS.mul_rank_size != 1:
    sample_one_device = int(len(image_list) / FLAGS.mul_rank_size)
    image_list = image_list[FLAGS.mul_device_id * sample_one_device:(FLAGS.mul_device_id+1)*sample_one_device]


def queue_thread():
    data_generator = dataset.get_batch(num_workers=cfg.num_readers,
                                           input_size=cfg.input_size,
                                           batch_size=cfg.batch_size_per_gpu * len(gpus),
                                           image_list=image_list)
    for i in range(cfg.max_steps):
         data = next(data_generator)
         q.put(data)
         #print(q.qsize())

q = queue.Queue(10)
t = threading.Thread(target=queue_thread, args=(), daemon=True)

def tower_loss(images, label_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)
    score_maps, geo_maps = tf.split(label_maps, num_or_size_splits=[1, 5], axis=-1)
    model_loss = loss.loss(score_maps, f_score, geo_maps, f_geometry, training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:4] * score_maps)
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:4] * score_maps)
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    """
    Average gradient accross mutli-gpus, this part almost not change in any codes.
    """
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


def broadcast_global_variables(root_rank, index):
    op_list = []
    for var in tf.global_variables():
        if "float" in var.dtype.name:
            inputs = [var]
            outputs = hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))
    return tf.group(op_list)


def main(argv=None):
    if not tf.gfile.Exists(cfg.checkpoint_path):
        tf.gfile.MkDir(cfg.checkpoint_path)
    else:
        if not cfg.restore:
            tf.gfile.DeleteRecursively(cfg.checkpoint_path)
            tf.gfile.MkDir(cfg.checkpoint_path)
    if FLAGS.mul_rank_size != 1:
        from hccl.split.api import set_split_strategy_by_size
        set_split_strategy_by_size([90, 10])
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_label_maps = tf.placeholder(tf.float32, shape=[None, None, None, 6], name='input_label_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    lr = cfg.learning_rate
    if FLAGS.mul_rank_size != 1:
        lr = lr * FLAGS.mul_rank_size
    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    #npu modify begin
    opt = tf.train.AdamOptimizer(learning_rate)
    #opt = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate))
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=100, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    
    if FLAGS.mul_rank_size != 1:
        opt = npu_distributed_optimizer_wrapper(opt)
        opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=True)
    else:
        opt = NPULossScaleOptimizer(opt, loss_scale_manager)
    
    
    # opt = NPULossScaleOptimizer(opt, loss_scale_manager)
    #npu modify end
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_label_maps_split = tf.split(input_label_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/cpu:0'):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                total_loss, model_loss = tower_loss(input_images_split[i],
                                                    input_label_maps_split[i],
                                                    input_training_masks_split[i],
                                                    reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(cfg.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(cfg.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()
    bcast_op = broadcast_global_variables(0, 1)
    # if cfg.pretrained_model_path is not None:
    #     variable_restore_op = slim.assign_from_checkpoint_fn(cfg.pretrained_model_path,
    #                                                          slim.get_trainable_variables(), ignore_missing_vars=True)
    #npu modify begin
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    if FLAGS.mul_rank_size != 1:
        custom_op.parameter_map['hcom_parallel'].b = True
    custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    
    #with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto(allow_soft_placement=True))) as sess:
    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
    #npu modify end
        if cfg.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(cfg.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.mul_rank_size != 1:
                sess.run(bcast_op)
            if cfg.pretrained_model_path is not None:
                variable_restore_op = slim.assign_from_checkpoint_fn(cfg.pretrained_model_path,
                                                                     slim.get_trainable_variables(),
                                                                     ignore_missing_vars=True)
                variable_restore_op(sess)
        t.start()
        TM_BEGIN('step_time')
        for step in range(cfg.max_steps):
            data = q.get()
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_label_maps: data[1],
                                                                                input_training_masks: data[2]})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                TM_END('step_time')
                TM_BEGIN('step_time')
                avg_time_per_step = TM_PICK('step_time') / 10.
                avg_examples_per_second = cfg.batch_size_per_gpu * len(gpus) / avg_time_per_step
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} '
                      'examples/second'.format(step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % cfg.save_checkpoint_steps == 0:
                saver.save(sess, cfg.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % cfg.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op],
                                              feed_dict={input_images: data[0],
                                                         input_label_maps: data[1],
                                                         input_training_masks: data[2]})
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    tf.app.run()
