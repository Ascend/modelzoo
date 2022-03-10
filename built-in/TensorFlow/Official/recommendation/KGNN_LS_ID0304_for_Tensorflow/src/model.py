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
from aggregators import SumAggregator, LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score
from npu_bridge.npu_init import *


@tf.custom_gradient
def gather_npu(params, indices):
    """
    :param params:
    :param indices:
    :return:
    """
    def grad(dy):
        """
        :param dy:
        :return:
        """
        params_shape = tf.shape(params, out_type=tf.int64)
        params_shape = tf.cast(params_shape, tf.int32)
        grad_gather = tf.unsorted_segment_sum(dy, indices, params_shape[0])
        return grad_gather, None

    return tf.gather(params, indices), grad


class KGNN_LS(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, offset):
        """
        :param args:
        :param n_user:
        :param n_entity:
        :param n_relation:
        :param adj_entity:
        :param adj_relation:
        :param offset:
        """
        self._parse_args(args, adj_entity, adj_relation, offset)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def embedding_lookup(self, params, ids):
        """
        :param params:
        :param ids:
        :return:
        """
        if self.n_iter == 1 and self.n_neighbor == 4 and self.dim == 8:
            return tf.nn.embedding_lookup(params, ids)
        else:
            return gather_npu(params, ids)

    def _parse_args(self, args, adj_entity, adj_relation, offset):
        """
        :param args:
        :param adj_entity:
        :param adj_relation:
        :param offset:
        :return:
        """
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        # LS regularization
        self.offset = offset
        self.ls_weight = args.ls_weight

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

    def _build_inputs(self):
        """
        :return:
        """
        self.hash_table = tf.placeholder(dtype=tf.float32, shape=[None], name='hash_table')
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        """
        :param n_user:
        :param n_entity:
        :param n_relation:
        :return:
        """
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGNN_LS.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGNN_LS.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGNN_LS.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = self.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # LS regularization
        self._build_label_smoothness_loss(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        """
        :param seeds:
        :return:
        """
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(gather_npu(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(gather_npu(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations):
        """
        :param entities:
        :param relations:
        :return:
        """
        aggregators = []  # store all aggregators
        entity_vectors = [self.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [self.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    # LS regularization
    def _build_label_smoothness_loss(self, entities, relations):
        """
        :param entities:
        :param relations:
        :return:
        """
        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        reset_masks = []  # True means the label of this item is reset to initial value during label propagation
        holdout_item_for_user = None
        self.uec = []
        self.il = []
        i = 0
        start = 0
        users = tf.expand_dims(self.user_indices, 1)
        for entities_per_iter in entities:

            # [batch_size, 1]

            # [batch_size, n_neighbor^i]
            user_entity_concat = users * self.offset + entities_per_iter

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            # [batch_size, n_neighbor^i]

            # initial_label = self.interaction_table.lookup(user_entity_concat)
            # initial_label = self.hash_table[i]
            end = start + self.batch_size * self.n_neighbor ** i
            tmp = self.hash_table[start:end]
            start = end
            initial_label = tf.reshape(tmp, [self.batch_size, -1])
            i += 1
            # self.uec.append(users)
            # self.il.append(entities_per_iter)

            holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)  # False if the item is held out
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)  # True if the entity is a labeled item
            reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # label initialization

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration

        # label propagation
        relation_vectors = [self.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        aggregator = LabelAggregator(self.batch_size, self.dim)
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                vector = aggregator(self_vectors=entity_labels[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_labels[hop + 1], [self.batch_size, -1, self.n_neighbor]),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], [self.batch_size, -1, self.n_neighbor, self.dim]),
                                    user_embeddings=self.user_embeddings,
                                    masks=reset_masks[hop])
                entity_labels_next_iter.append(vector)
            entity_labels = entity_labels_next_iter

        self.predicted_labels = tf.squeeze(entity_labels[0], axis=-1)

    def _build_train(self):
        """
        :return:
        """
        # base loss
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # L2 loss
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # LS loss
        self.ls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.predicted_labels))
        self.loss += self.ls_weight * self.ls_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        """
        :param sess:
        :param feed_dict:
        :return:
        """
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        """
        :param sess:
        :param feed_dict:
        :return:
        """
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        """
        :param sess:
        :param feed_dict:
        :return:
        """
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
