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

import _init_paths
from graphsage.models import SampleAndAggregate, SAGEInfo
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from graphsage.logger import setup_logger

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 20, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. '
                                        'Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_integer('save_embeddings', 1, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', 'outputs/unsup', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 10 ** 10, "how often to run a validation minibatch.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")
flags.DEFINE_integer('iterations_per_loop', 100, 'Number of iterations per loop')

# device
flags.DEFINE_string('device', 'npu', 'device to run the model. Can be cpu, gpu, npu. Default npu')
flags.DEFINE_string('device_ids', '0', "which device to use.")
flags.DEFINE_string('enable_mix_precision', 'on', 'on or off')
flags.DEFINE_integer('rank_size', 1, 'rank size')


def log_dir():
    log_dir = FLAGS.base_log_dir + "/unsup-{}p".format(FLAGS.rank_size)
    log_dir += "/{model:s}/".format(
        model=FLAGS.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


def save_val_embeddings(sess, model, placeholders, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(placeholders, size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                            feed_dict=feed_dict_val)
        # ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i, :])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy", val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str, nodes)))


def train(train_data, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if features is not None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    
    to_save = FLAGS.rank_size == 1 or FLAGS.device_ids == '0'
    logger = setup_logger("GraphSAGE", log_dir() + 'train_log.txt', to_save)

    minibatch = EdgeMinibatchIterator(G,
                                      id_map,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      num_neg_samples=FLAGS.neg_sample_size,
                                      context_pairs=context_pairs)

    steps_per_epoch = len(minibatch.train_edges) // FLAGS.batch_size // FLAGS.rank_size
    if FLAGS.max_total_steps == 10 ** 10:
        FLAGS.max_total_steps = steps_per_epoch * FLAGS.epochs
    if FLAGS.device == 'npu':
        # Iterations loop for NPU only
        steps_per_epoch = steps_per_epoch // FLAGS.iterations_per_loop
        FLAGS.max_total_steps = FLAGS.max_total_steps // FLAGS.iterations_per_loop
    else:
        FLAGS.iterations_per_loop = 1

    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    dataset = tf.data.Dataset.from_tensor_slices(np.array(minibatch.train_edges))
    dataset = dataset.shuffle(buffer_size=1000).batch(FLAGS.batch_size, drop_remainder=True).repeat()
    if FLAGS.device == 'npu' and FLAGS.rank_size > 1:
        rank_id = int(FLAGS.device_ids)
        logger.info('train ranksize = {}, rankid = {}'.format(FLAGS.rank_size, rank_id))
        dataset = dataset.shard(FLAGS.rank_size, rank_id)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    placeholders = dict(batch1=next_element[:, 0], batch2=next_element[:, 1])

    # Create model
    if FLAGS.model in [ 'mean', 'meanpool', 'maxpool']:
        concat = True
    elif FLAGS.model in ['gcn']:
        concat = False
    else:
        raise Exception('Error: model name unrecognized.')
    
    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                   SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

    model = SampleAndAggregate(placeholders,
                               features,
                               adj_info,
                               minibatch.deg,
                               layer_infos=layer_infos,
                               aggregator_type=FLAGS.model,
                               model_size=FLAGS.model_size,
                               identity_dim=FLAGS.identity_dim,
                               concat=concat,
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
            logger.info('Turn npu mix precision mode on for better performance')
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    # Initialize session
    sess = tf.Session(config=config)
    
    # Save graph
    if to_save:
        tf.train.write_graph(sess.graph, log_dir(), 'graph.pbtxt')

    # Init variables
    if FLAGS.device == 'npu':
        if FLAGS.iterations_per_loop > 1:
            from npu_bridge.estimator.npu import util
            model.opt_op = util.set_iteration_per_loop(sess, model.opt_op, FLAGS.iterations_per_loop)
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Train model
    avg_time = 0.0

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    sess.run(iterator.initializer)

    for it in range(FLAGS.max_total_steps):
        # Training step
        outs = sess.run([model.opt_op, model.loss])
        train_cost = outs[1]

        if it % max(FLAGS.print_every // FLAGS.iterations_per_loop, 1) == 0:
            logger.info(
                "step={:d}, epoch={:.1f}, loss={:.3f}".
                    format(it, it / steps_per_epoch, train_cost) + \
                (", batch_time={:.5f} s/step".format(avg_time) if avg_time > 0 else ""))
    
        st_it = max(100 // FLAGS.iterations_per_loop, 1)
        if it > st_it:
            avg_time = (avg_time * (it - st_it - 1) * FLAGS.iterations_per_loop + time.time() - t1) / (it - st_it) / FLAGS.iterations_per_loop
        t1 = time.time()

    logger.info("Optimization Finished!")
    logger.info("Runtime: {:5f} ms / step, {:5f} minutes totally".format(avg_time * 1000, avg_time * FLAGS.max_total_steps * FLAGS.iterations_per_loop / 60))
    
    if FLAGS.save_embeddings:
        if to_save:
            sess.run(val_adj_info.op)
            save_val_embeddings(sess, model, placeholders, minibatch, FLAGS.batch_size, log_dir())

    
def main(argv=None):
    if FLAGS.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_ids)
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
        raise KeyError('Unknown device type')

    FLAGS.batch_size = FLAGS.batch_size // FLAGS.rank_size
    FLAGS.neg_sample_size = min(FLAGS.neg_sample_size // FLAGS.rank_size, 5)

    train_data = load_data(FLAGS.train_prefix, load_walks=True)
    train(train_data)
    

if __name__ == '__main__':
    tf.app.run()
