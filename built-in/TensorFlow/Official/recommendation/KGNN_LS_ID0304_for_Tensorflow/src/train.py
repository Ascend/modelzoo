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

import tensorflow as tf
import numpy as np
from model import KGNN_LS
from npu_bridge.npu_init import *
import time
from datetime import timedelta
import pickle


def get_time_dif(start_time, i=0):
    """
    :param start_time:
    :param i:
    :return:
    """
    end_time = time.time() * 1000
    time_dif = end_time - start_time * 1000
    if i != 0:
        print("step Time usage:", timedelta(milliseconds=int(round((end_time - start_time * 1000) / i))))
    return timedelta(milliseconds=int(round(time_dif)))


def train(args, data, show_loss, show_topk, seed):
    """
    :param args:
    :param data:
    :param show_loss:
    :param show_topk:
    :param seed:
    :return:
    """
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    # offset = get_interaction_table(train_data, n_entity)
    offset = 10 ** len(str(n_entity))
    model = KGNN_LS(args, n_user, n_entity, n_relation, adj_entity, adj_relation, offset)

    # Training and validation loop
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["use_off_line"].b = True

    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        # tf.io.write_graph(sess.graph_def, 'dump', 'train.pbtxt')
        # interaction_table.init.run()
        RAllStep = 0
        RAllTime = 0.0
        best_eval_auc = 0.0
        best_epoch = 0
        for step in range(args.n_epochs):

            # training

            # np.random.shuffle(train_data)
            while not os.path.exists('newdata/e%s.pkl' % step):
                time.sleep(0.1)
            with open('newdata/e%s.pkl' % step, 'rb') as f:
                dataset = pickle.load(f)
            train_data = dataset["train_data"]
            hash = dataset["hash"]
            start = 0
            i = 0
            # skip the last incomplete minibatch if its size < batch size
            print("=" * 20)
            start_time = time.time()
            while start + args.batch_size <= train_data.shape[0]:
                startTime11 = time.time() * 1000
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size, hash[i]))
                i += 1
                # print("step Time usage:", get_time_dif(start_time2))
                # exit()
                # if i==1:
                #    break
                #    exit()

                start += args.batch_size
                if show_loss:
                    print('loss start and loss print: %d %d' % (start, loss))
                endTime11 = time.time() * 1000
                difftime = endTime11 - startTime11
                RAllStep += 1
                RAllTime += difftime
            print("epoch Time usage:", get_time_dif(start_time, i))
            # exit()
            # CTR evaluation

            start_time = time.time()
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            print("train test Time usage:", get_time_dif(start_time))

            start_time = time.time()
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            print("eval test Time usage:", get_time_dif(start_time))

            # start_time = time.time()
            # test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)
            # print("test test Time usage:", get_time_dif(start_time))

            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1))
            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('top-K precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('top-K recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')

            # save model
            save_path = args.save_dir
            if eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                base_epoch = step
                saver.save(sess=sess, save_path=save_path)
        print('best_epoch: %d best_eval_auc: %.4f' % (best_epoch, best_eval_auc))
        print('avrg steps per second: ' + str((RAllStep / RAllTime) * 1000))
        print('average FPS is : ' + str((RAllStep / RAllTime) * 1000 * args.batch_size))

# interaction_table is used for fetching user-item interaction label in LS regularization
# key: user_id * 10^offset + item_id
# value: y_{user_id, item_id}
def get_interaction_table(train_data, n_entity):
    """
    :param train_data:
    :param n_entity:
    :return:
    """
    offset = len(str(n_entity))
    offset = 10 ** offset
    # keys = train_data[:, 0] * offset + train_data[:, 1]

    # print("4",train_data[:, 0],train_data[:, 1])
    # keys = keys.astype(np.int64)
    # values = train_data[:, 2].astype(np.float32)
    # interaction_table =None
    # interaction_table = tf.contrib.lookup.HashTable(
    #    tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return offset


def topk_settings(show_topk, train_data, test_data, n_item):
    """
    :param show_topk:
    :param train_data:
    :param test_data:
    :param n_item:
    :return:
    """
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end, hash=[]):
    """
    :param model:
    :param data:
    :param start:
    :param end:
    :param hash:
    :return:
    """
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.hash_table: hash}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    """
    :param sess:
    :param model:
    :param data:
    :param batch_size:
    :return:
    """
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    """
    :param sess:
    :param model:
    :param user_list:
    :param train_record:
    :param test_record:
    :param item_set:
    :param k_list:
    :param batch_size:
    :return:
    """
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    """
    :param data:
    :param is_train:
    :return:
    """
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
