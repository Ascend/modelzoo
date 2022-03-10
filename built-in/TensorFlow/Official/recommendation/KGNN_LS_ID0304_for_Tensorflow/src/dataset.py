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
import time
import time
from data_loader import load_data
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--seed', type=int, default=555, help='seed')
args = parser.parse_args()
np.random.seed(args.seed)
data = np.load("newdata/data.npy", allow_pickle=True)

n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
train_data, eval_data, test_data = data[4], data[5], data[6]
adj_entity, adj_relation = data[7], data[8]


# aa = train_data[:,2].astype(np.float32)
# train_data[:,2] = aa
# print(type(train_data[0,0]),type(train_data[0,1]),type(train_data[0,2]),train_data[0,2],aa[0])
# exit()

def get_neighbors_(seeds):
    """
    :param seeds:
    :return:
    """
    seeds = tf.expand_dims(seeds, axis=1)
    entities = [seeds]
    relations = []
    for i in range(args.n_iter):
        neighbor_entities = tf.reshape(tf.gather(adj_entity, entities[i]), [args.batch_size, -1])
        entities.append(neighbor_entities)
    return entities


def get_interaction_table(train_data, n_entity):
    """
    :param train_data:
    :param n_entity:
    :return:
    """
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)
    interaction = None
    interaction = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return interaction, offset


interaction_table, offset = get_interaction_table(train_data, n_entity)
item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
entities = get_neighbors_(item_indices)
b = []
users = tf.expand_dims(item_indices, 1)
for entities_per_iter in entities:
    user_entity_concat = users * offset + entities_per_iter
    initial_label = interaction_table.lookup(user_entity_concat)
    b.append(initial_label)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    interaction_table.init.run()

    o = 0

    for i in range(args.n_epochs):
        # aa= time.time()
        train_data = np.random.permutation(train_data)
        # print("shuffle",time.time()-aa)
        t = []
        # aa= time.time()
        start = 0
        while start + args.batch_size <= train_data.shape[0]:
            # aa= time.time()
            ret = sess.run(b, feed_dict={item_indices: train_data[start:start + args.batch_size, 1]})
            t.append(np.concatenate([x.ravel() for x in ret]))
            # print(time.time()-aa)
            start += args.batch_size
        # print("epoch",time.time()-aa)
        # aa= time.time()
        sa = {"train_data": train_data, "hash": t}
        f = open('newdata/e%s.pkl_' % i, 'wb')
        pickle.dump(sa, f)
        f.close()
        os.rename('newdata/e%s.pkl_' % i, 'newdata/e%s.pkl' % i)
        # print("save",time.time()-aa)
