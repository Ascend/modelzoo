#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
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
from tensorflow.core.protobuf import config_pb2
import tensorflow as tf
import coms.utils as utils
import coms.pre_process as pre_pro
import coms.coms as coms
from coms.tfrecords import get_tfrecords
import net.GoogLeNet.InceptionV3 as InceptionV3
import coms.learning_rate as LR_Tools
import time
import cv2
import numpy as np
import os
tf.app.flags.DEFINE_integer('epoch_num', 1, '')
tf.app.flags.DEFINE_string('NPU_DEVICE_INDEX', '0', '')
tf.app.flags.DEFINE_integer('npu_nums', 1, '')
tf.app.flags.DEFINE_string('precision_mode', 'allow_fp32_to_fp16', '')
FLAGS = tf.app.flags.FLAGS

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

def run():
    model_dir = ''
    logdir = ''
    img_prob = [299, 299, 3]
    num_cls = 10
    is_train = True
    is_load_model = False
    is_stop_test_eval = True
    BATCH_SIZE = 32
    EPOCH_NUM = FLAGS.epoch_num
    ITER_NUM = 1500
    LEARNING_RATE_VAL = 0.001
    if utils.isLinuxSys():
        logdir = './result/' + FLAGS.NPU_DEVICE_INDEX+ '/log'
        model_dir = './result/' + FLAGS.NPU_DEVICE_INDEX + '/model'
    else:
        model_dir = 'D:\\DataSets\\cifar\\cifar\\model_flie\\inceptionv3'
        logdir = 'D:\\DataSets\\cifar\\cifar\\logs\\train\\inceptionv3'
    if is_train:
        #train_img_batch, train_label_batch = pre_pro.get_cifar10_batch(is_train=True, batch_size=BATCH_SIZE, num_cls=num_cls, img_prob=img_prob)
        #test_img_batch, test_label_batch = pre_pro.get_cifar10_batch(is_train=False, batch_size=BATCH_SIZE, num_cls=num_cls, img_prob=img_prob)
        ds_train = pre_pro.get_cifar10_batch(is_train=True, batch_size=BATCH_SIZE, num_cls=num_cls, img_prob=img_prob)
        iter_train = ds_train.make_initializable_iterator()

        ds_test = pre_pro.get_cifar10_batch(is_train=False, batch_size=BATCH_SIZE, num_cls=num_cls, img_prob=img_prob)
        iter_test = ds_test.make_initializable_iterator()



    inputs = tf.placeholder(tf.float32, [None, img_prob[0], img_prob[1], img_prob[2]])
    labels = tf.placeholder(tf.float32, [None, num_cls])
    is_training = tf.placeholder(tf.bool)
    LEARNING_RATE = tf.placeholder(tf.float32)
    calc_lr = LR_Tools.CLR_EXP_RANGE()
    logits = InceptionV3.V3_slim(inputs, num_cls, is_training=is_training)
    train_loss = coms.loss(logits, labels)
    train_optim = coms.optimizer_bn(lr=LEARNING_RATE, loss=train_loss)
    train_eval = coms.evaluation(logits, labels)
    saver = tf.train.Saver(max_to_keep=4)
    max_acc = 0.0
    if True:
        config = tf.ConfigProto(allow_soft_placement=True)
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True

        config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF 
    if FLAGS.precision_mode == "allow_mix_precision":
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    # for npu
    # config = tf.ConfigProto()
    # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name = "NpuOptimizer"
    # custom_op.parameter_map["use_off_line"].b = True
    # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
     # for npu
    broad_op = broadcast_global_variables(0, 1)
    with tf.Session(config=config) as sess:
        if utils.isHasGpu():
            dev = '/gpu:0'
        else:
            dev = '/cpu:0'
        with tf.device('/cpu:0'):
            sess.run(tf.global_variables_initializer())
            sess.run(iter_train.initializer)
            sess.run(iter_test.initializer)
            next_iter_train = iter_train.get_next()
            next_iter_test = iter_test.get_next()
#            coord = tf.train.Coordinator()
#            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                if is_train:
                    if is_load_model:
                        ckpt = tf.train.get_checkpoint_state(model_dir)
                        if (ckpt and ckpt.model_checkpoint_path):
                            saver.restore(sess, ckpt.model_checkpoint_path)
                            print('model load successful ...')
                        else:
                            print('model load failed ...')
                            return
                    n_time = time.strftime('%Y-%m-%d %H-%M', time.localtime())
                    logdir = os.path.join(logdir, n_time)
                    writer = tf.summary.FileWriter(logdir, sess.graph)
                    for epoch in range(EPOCH_NUM):
#                        if coord.should_stop():
#                            print('coord should stop ...')
#                            break
                        duration = 0
                        for step in range(1, (ITER_NUM + 1)):
                            if FLAGS.npu_nums == 8:
                                sess.run(broad_op)
#                            if coord.should_stop():
#                                print('coord should stop ...')
#                                break
                            start_time = time.time()
                            LEARNING_RATE_VAL = calc_lr.calc_lr(step, ITER_NUM, 0.001, 0.01, gamma=0.9998)
                            # for npu  
                            #(batch_train_img, batch_train_label) = sess.run([train_img_batch, train_label_batch])
                            batch_train_img, batch_train_label = sess.run(next_iter_train)                            
                            # for npu

                            (_, batch_train_loss, batch_train_acc) = sess.run([train_optim, train_loss, train_eval], feed_dict={inputs: batch_train_img, labels: batch_train_label, LEARNING_RATE: LEARNING_RATE_VAL, is_training: is_train})
                            global_step = int((((epoch * ITER_NUM) + step) + 1))
                            duration += (time.time() - start_time)
                            print(('epoch %d , step %d train end ,loss is : %f ,accuracy is %f, time cust is %f ... ...' % (epoch, step, batch_train_loss, batch_train_acc, duration)))
                            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=batch_train_loss), tf.Summary.Value(tag='train_batch_accuracy', simple_value=batch_train_acc), tf.Summary.Value(tag='learning_rate', simple_value=LEARNING_RATE_VAL)])
                            writer.add_summary(train_summary, global_step)
                            writer.flush()
                            duration = 0
                            if is_stop_test_eval:
                                if (not is_load_model):
                                    if (epoch < 3):
                                        continue
                            if ((step % 100) == 0):
                                print('test sets evaluation start ...')
                                ac_iter = int((10000 / BATCH_SIZE))
                                ac_sum = 0.0
                                loss_sum = 0.0
                                for ac_count in range(ac_iter):
                                    # for npu
                                    #(batch_test_img, batch_test_label) = sess.run([test_img_batch, test_label_batch])
                                    batch_test_img, batch_test_label = sess.run(next_iter_test)                                    
                                    # for npu
                                    (test_loss, test_accuracy) = sess.run([train_loss, train_eval], feed_dict={inputs: batch_test_img, labels: batch_test_label, is_training: False})
                                    ac_sum += test_accuracy
                                    loss_sum += test_loss
                                ac_mean = (ac_sum / ac_iter)
                                loss_mean = (loss_sum / ac_iter)
                                print('epoch {} , step {} , accuracy is {}'.format(str(epoch), str(step), str(ac_mean)))
                                test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss_mean), tf.Summary.Value(tag='test_accuracy', simple_value=ac_mean)])
                                writer.add_summary(test_summary, global_step=global_step)
                                writer.flush()
                                if (ac_mean >= max_acc):
                                    max_acc = ac_mean
                                    saver.save(sess, ((model_dir + '/') + 'cifar10_{}_step_{}.ckpt'.format(str(epoch), str(step))), global_step=step)
                                    print('max accuracy has reaching ,save model successful ...')
                    print('train network task was run over')
                else:
                    model_file = tf.train.latest_checkpoint(model_dir)
                    saver.restore(sess, model_file)
                    cls_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                    for i in range(1, 11):
                        name = (str(i) + '.jpg')
                        img = cv2.imread(name)
                        img = cv2.resize(img, (299, 299))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = (img / 255.0)
                        img = np.array([img])
                        res = sess.run(logits, feed_dict={inputs: img, is_training: False})
                        print(('{}.jpg detect result is : '.format(str(i)) + cls_list[np.argmax(res)]))
            except tf.errors.OutOfRangeError:
                print('done training -- opoch files run out of ...')
#            finally:
#                coord.request_stop()
#            coord.join(threads)
            sess.close()
if (__name__ == '__main__'):
    np.set_printoptions(suppress=True)
    run()
