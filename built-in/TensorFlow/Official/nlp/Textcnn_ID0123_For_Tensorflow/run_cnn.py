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

from __future__ import print_function
from npu_bridge.npu_init import *
#from npu_bridge import *
import os
import sys
import time
from datetime import timedelta
import pickle
import numpy as np
import tensorflow as tf
from sklearn import metrics
from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', dest='save_dir', default='checkpoints/textcnn')
parser.add_argument('--data_path', dest='data_path', default='./data/cnews', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_fp32_to_fp16', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--autotune', dest='autotune', default='False', help='whether to enable autotune, default is False')
parser.add_argument('--npu_loss_scale', dest='npu_loss_scale', type=int, default=1)
parser.add_argument('--mode', dest='mode', default='train', choices=('train', 'test', 'train_and_eval'))
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=10)
args = parser.parse_args()

base_dir = args.data_path
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = args.save_dir
save_path = os.path.join(save_dir, 'best_validation')

def get_time_dif(start_time):
    '获取已使用时间'
    end_time = time.time()
    time_dif = (end_time - start_time)
    return timedelta(seconds=int(round(time_dif))), time_dif

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x,y):
    """评估在某一数据上的准确率和损失"""    
    total_loss = 0.0
    total_acc = 0.0
    data_len = len(x)
    batch_train = batch_iter_(x, y,256)
    for x_batch, y_batch in batch_train:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        (loss, acc) = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += (loss * batch_len)
        total_acc += (acc * batch_len)
    return ((total_loss / data_len), (total_acc / data_len))
class data_load(object):
    def __init__(self, sess,x,y,is_train=True):
        
        with tf.device('/cpu:0'):
            self.x = x
            self.y = y
            self.x_ = tf.placeholder(self.x.dtype, self.x.shape)
            self.y_ = tf.placeholder(self.y.dtype, self.y.shape)
            self.sess = sess
            dataset = tf.data.Dataset.from_tensor_slices((self.x_, self.y_))

            if is_train:
                dataset = dataset.shuffle(len(self.x))
                dataset = dataset.repeat()
                dataset = dataset.batch(len(self.x))
            else:
                dataset = dataset.batch(len(self.x))
            
            dataset = dataset.prefetch(2)
            self.iterator = dataset.make_initializable_iterator()
            self.next = self.iterator.get_next()
            self.sess.run(self.iterator.initializer, feed_dict={self.x_: self.x,self.y_: self.y})
        
    def replay(self):
        self.sess.run(self.iterator.initializer, feed_dict={self.x_: self.x,self.y_: self.y})
    
    
def batch_iter_(x, y, batch_size=64):
        data_len = len(x)
        
        num_batch = int((data_len - 1) / batch_size) + 1
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x[start_id:end_id], y[start_id:end_id]
def train():
    print('Configuring TensorBoard and Saver...')
    tensorboard_dir = 'tensorboard/textcnn'
    if (not os.path.exists(tensorboard_dir)):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    print('Loading training and validation data...')
    start_time = time.time()
    (x_train, y_train) = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    (x_val, y_val) = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    ############################ modify for run on npu ###############################
    from npu_bridge.estimator import npu_ops
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True # 必须显示开启，在昇腾AI处理器执行训练
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    if args.data_dump_flag.strip() == "True":
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(args.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if args.over_dump.strip() == "True":
        # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
        # enable_dump_debug：是否开启溢出检测功能
        custom_op.parameter_map["enable_dump_debug"].b = True
        # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    if args.profiling.strip() == "True":
        custom_op.parameter_map["profiling_mode"].b = False
        profilingvalue = (
                '{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}' % (
            args.profiling_dump_path))
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profilingvalue)
    ############################ modify for run on npu ###############################
    print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC Finish")
    session = tf.Session(config=sess_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    train_len = len(x_train)
    val_len = len(x_val)
    train_data = data_load(session,x_train,y_train)
    val = data_load(session,x_val,y_val,False)
    x_v, y_v = session.run(val.next)
    tf.io.write_graph(session.graph_def, 'checkpoints', 'train.pbtxt')
    print('Training and evaluating...')
    start_time = time.time()
    data_time = 0
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 10000
    total_feed = 0
    total_summary = 0
    total_val = 0
    total_save = 0
    total_train = 0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', (epoch + 1))
        x, y = session.run(train_data.next)
        batch_train = batch_iter_(x, y, config.batch_size)
        for (x_batch, y_batch) in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            #if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                #s = session.run(merged_summary, feed_dict=feed_dict)
                #writer.add_summary(s, total_batch)
            if ((total_batch % config.print_per_batch) == 0):
                feed_dict[model.keep_prob] = 1.0
                (loss_train, acc_train) = session.run([model.loss, model.acc], feed_dict=feed_dict)
                (loss_val, acc_val) = evaluate(session, x_v, y_v)
                if (acc_val > best_acc_val):
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''
                time_dif, time_sec = get_time_dif(start_time)
                msg = ('Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6} ({7})')
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str, time_sec))
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)
            #time_dif = get_time_dif(start_time)
            #print("step:%d, time:%s"%(total_batch, time_dif))
            total_batch += 1
            if ((total_batch - last_improved) > require_improvement):
                # 验证集正确率长期不提升，提前结束训练
                print('No optimization for a long time, auto-stopping...')
                flag = True
                break  # 跳出循环
        if flag:
            break

def test():
    print('Loading test data...')
    
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    session = tf.Session(config=sess_config)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    start_time = time.time()
    print('Testing...')
    (loss_test, acc_test) = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    batch_size = 256
    data_len = len(x_test)
    num_batch = (int(((data_len - 1) / batch_size)) + 1)
    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)
    for i in range(num_batch):
        start_id = (i * batch_size)
        end_id = min(((i + 1) * batch_size), data_len)
        feed_dict = {model.input_x: x_test[start_id:end_id], model.keep_prob: 1.0}
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    print('Precision, Recall and F1-Score...')
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)
if (__name__ == '__main__'):
    print('Configuring CNN model...')
    config = TCNNConfig()
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.npu_loss_scale = args.npu_loss_scale
    if (not os.path.exists(vocab_dir)):
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    (categories, cat_to_id) = read_category()
    (words, word_to_id) = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)
    if (args.mode == 'train'):
        train()
    elif (args.mode == 'test'):
        test()
    else:
        train()
        test()