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

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
#from sklearn import metrics
#import pickle as pkl

from utils import *
from models import GNN, MLP

import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'mr', 'Dataset string.')  # 'mr','ohsumed','R8','R52'
flags.DEFINE_string('data_url', './data', 'Path to dataset directory.')
flags.DEFINE_string('train_url', './output', 'Path to output directory.')
flags.DEFINE_string('model', 'gnn', 'Model string.') 
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 1024, 'Size of batches per epoch.')
flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
flags.DEFINE_integer('hidden', 96, 'Number of units in hidden layer.') # 32, 64, 96, 128
flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.') # 5e-4
flags.DEFINE_integer('early_stopping', -1, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # Not used


# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(FLAGS.dataset, FLAGS.data_url)

max_length = max([len(i) for i in train_adj] + [len(j) for j in val_adj] + [len(k) for k in test_adj])

# Some preprocessing
print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj, max_length)
train_feature = preprocess_features(train_feature, max_length)
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj, max_length)
val_feature = preprocess_features(val_feature, max_length)
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj, max_length)
test_feature = preprocess_features(test_feature, max_length)


if FLAGS.model == 'gnn':
    # support = [preprocess_adj(adj)]
    # num_supports = 1
    model_func = GNN
elif FLAGS.model == 'gcn_cheby': # not used
    # support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GNN
elif FLAGS.model == 'dense': # not used
    # support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None, max_length, max_length)),
    'features': tf.placeholder(tf.float32, shape=(None, max_length, FLAGS.input_dim)),
    'mask': tf.placeholder(tf.float32, shape=(None, max_length, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, train_y.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}


# label smoothing
# label_smoothing = 0.1
# num_classes = y_train.shape[1]
# y_train = (1.0 - label_smoothing) * y_train + label_smoothing / num_classes


# Create model
model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('logs/', sess.graph)

# Define model evaluation function
def evaluate(features, support, mask, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


cost_val = []
best_val = 0
best_epoch = 0
best_acc = 0
best_cost = 0
test_doc_embeddings = None
preds = None
labels = None
#tf.summary.scalar('loss', model.loss)
#tf.summary.scalar('accuracy', model.accuracy)
#summary_op = tf.summary.merge_all()

print('train start...')
# Train model
# Initialize session
with tf.Session(config=config) as sess:
    # Init variables
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    #train_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.train_url, "train"), graph=sess.graph)
    #test_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.train_url, "test"), graph=sess.graph)

    for epoch in range(FLAGS.epochs):
        t = time.time()
        
        # Training step
        indices = np.arange(0, len(train_y))
        np.random.shuffle(indices)
    
        train_loss, train_acc = 0, 0
        for start in range(0, len(train_y), FLAGS.batch_size):
            end = start + FLAGS.batch_size
            idx = indices[start:end]
            # Construct feed dictionary
            feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_mask[idx], train_y[idx], placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            train_loss += outs[1]*len(idx)
            train_acc += outs[2]*len(idx)
        train_loss /= len(train_y)
        train_acc /= len(train_y)
        #train_writer.add_summary(outs[3], epoch)

        # Validation
        val_cost, val_acc, val_duration, _, _, _ = evaluate(val_feature, val_adj, val_mask, val_y, placeholders)
        cost_val.append(val_cost)
    
        # Test
        test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(test_feature, test_adj, test_mask, test_y, placeholders)
        #test_writer.add_summary(summary, epoch)

        #if val_acc >= best_val:
        #    best_val = val_acc
        #    best_epoch = epoch
        best_acc = test_acc
        best_cost = test_cost
        test_doc_embeddings = embeddings
        preds = pred
        #test_writer.add_summary(summary=summary, global_step=epoch)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
              "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(test_acc), 
              "time=", "{:.5f}".format(time.time() - t))

        if FLAGS.early_stopping > 0 and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    #train_writer.close()
    #test_writer.close()
    print("Optimization Finished!")

    # Best results
    #print('Best epoch:', best_epoch)
    print("Test set results:", "cost=", "{:.5f}".format(best_cost),
          "accuracy=", "{:.5f}".format(best_acc))

    #print("Test Precision, Recall and F1-Score...")
    #print(metrics.classification_report(labels, preds, digits=4))
    #print("Macro average Test Precision, Recall and F1-Score...")
    #print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
    #print("Micro average Test Precision, Recall and F1-Score...")
    #print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))

'''
# For visualization
doc_vectors = []
for i in range(len(test_doc_embeddings)):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append(str(np.argmax(test_y[i])) + ' ' + doc_vector_str)

doc_embeddings_str = '\n'.join(doc_vectors)
with open('data/' + FLAGS.dataset + '_doc_vectors.txt', 'w'):
    f.write(doc_embeddings_str)
'''
