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

from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
from sklearn import metrics
import _init_paths
from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from graphsage.logger import setup_logger
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', True, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', 'outputs/sup', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 10 ** 10, "how often to run a validation minibatch.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")
flags.DEFINE_integer('iterations_per_loop', 1, 'Number of iterations per loop')

# device
flags.DEFINE_string('device', 'npu', 'device to run the model. Can be cpu, gpu, npu. Default npu')
flags.DEFINE_string('device_ids', '0', "which device to use.")
flags.DEFINE_string('enable_mix_precision', 'on', "on or off")
flags.DEFINE_integer('rank_size', 1, 'rank size')


def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-{}p".format(FLAGS.rank_size)
    log_dir += "/{model:s}/".format(
        model=FLAGS.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, placeholders, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _, valid_num = minibatch_iter.incremental_node_val_feed_dict(
            placeholders, size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0][-valid_num:])
        labels.append(batch_labels[-valid_num:])
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def inference(train_data, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if features is not None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None

    to_save = FLAGS.rank_size == 1 or FLAGS.device_ids == '0'
    logger = setup_logger("GraphSAGE", log_dir() + 'train_log.txt', to_save)

    minibatch = NodeMinibatchIterator(G,
                                      id_map,
                                      class_map,
                                      num_classes,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      context_pairs=context_pairs)

    # steps_per_epoch = len(minibatch.train_nodes) // FLAGS.batch_size // FLAGS.rank_size
    # if FLAGS.max_total_steps == 10 ** 10:
    #     FLAGS.max_total_steps = steps_per_epoch * FLAGS.epochs
    # if FLAGS.device == 'npu':
    #     # Iterations loop for NPU only
    #     steps_per_epoch = steps_per_epoch // FLAGS.iterations_per_loop
    #     FLAGS.max_total_steps = FLAGS.max_total_steps // FLAGS.iterations_per_loop
    # else:
    #     FLAGS.iterations_per_loop = 1

    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    train_nodes = np.array(minibatch.train_nodes, dtype=np.int64)
    train_label = np.array([minibatch.label_map[n] for n in range(len(train_nodes))], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((train_nodes, train_label))
    dataset = dataset.shuffle(buffer_size=1000).batch(FLAGS.batch_size, drop_remainder=True).repeat()
    if FLAGS.device == 'npu' and FLAGS.rank_size > 1:
        rank_id = int(FLAGS.device_ids)
        logger.info('train ranksize = {}, rankid = {}'.format(FLAGS.rank_size, rank_id))
        dataset = dataset.shard(FLAGS.rank_size, rank_id)

    placeholders = dict(labels=tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
                        batch=tf.placeholder(tf.int32, shape=(None), name='batch1'))

    # Create model
    if FLAGS.model in ['mean', 'meanpool', 'maxpool']:
        concat = True
    elif FLAGS.model in ['gcn']:
        concat = False
    else:
        raise Exception('Error: model name unrecognized.')

    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                   SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

    model = SupervisedGraphsage(num_classes,
                                placeholders,
                                features,
                                adj_info,
                                minibatch.deg,
                                layer_infos=layer_infos,
                                aggregator_type=FLAGS.model,
                                model_size=FLAGS.model_size,
                                concat=concat,
                                sigmoid_loss=FLAGS.sigmoid,
                                identity_dim=FLAGS.identity_dim,
                                logging=True,
                                enable_gpu_mix_precision=FLAGS.device == 'gpu' and FLAGS.enable_mix_precision == 'on',
                                enable_npu_distribution=FLAGS.device == 'npu' and FLAGS.rank_size > 1)

    # Session config
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    if FLAGS.device == 'gpu':
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
    elif FLAGS.device == 'npu':
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = False
        custom_op.parameter_map["mix_compile_mode"].b = False
        custom_op.parameter_map["use_off_line"].b = True
        if FLAGS.iterations_per_loop > 1:
            custom_op.parameter_map["iterations_per_loop"].i = FLAGS.iterations_per_loop
        if FLAGS.enable_mix_precision != 'on':
            logger.info('Automatically turn npu mix precision mode on for better runtime performance')
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    # Initialize session
    sess = tf.Session(config=config)

    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    # Config npu
    if FLAGS.device == 'npu':
        if FLAGS.iterations_per_loop > 1:
            from npu_bridge.estimator.npu import util
            model.opt_op = util.set_iteration_per_loop(sess, model.opt_op, FLAGS.iterations_per_loop)

    saver = tf.train.Saver()
    ckpt_dir = './model.ckpt'
    saver.restore(sess, ckpt_dir)

    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['Sigmoid'])
    with tf.gfile.FastGFile('supervisedGraphSage.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())



def main(argv=None):
    if FLAGS.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_ids)
    elif FLAGS.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    elif FLAGS.device == 'npu':
        from npu_bridge.estimator import npu_ops
        from npu_bridge.estimator.npu.npu_config import NPURunConfig
        from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
        from npu_bridge.estimator.npu.npu_optimizer import allreduce
        from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
        from npu_bridge.hccl import hccl_ops
    else:
        raise KeyError
    
    train_data = load_data(FLAGS.train_prefix)
    inference(train_data)



if __name__ == '__main__':
    tf.app.run()
