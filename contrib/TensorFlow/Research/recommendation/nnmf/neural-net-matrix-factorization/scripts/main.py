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

#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Trains/evaluates NNMF models."""
# Standard modules
import argparse, json, time, os
# Third party modules
import tensorflow as tf
import pandas as pd
import numpy as np
# Package modules
from nnmf.models import NNMF, SVINNMF
from nnmf.utils import chunk_df

#os.environ['DUMP_GE_GRAPH']='2'
#os.environ['PRINT_MODEL']='1'
#os.environ['EXPERIMENTAL_DYNAMIC_PARTITION']="1"
#os.environ['ASCEND_GLOBAL_LOG_LEVEL']='3'
#os.environ['ASCEND_GLOBAL_EVENT_ENABLE']='0'
#os.environ['ASCEND_SLOG_PRINT_TO_STDOUT']='1'
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import allreduce
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

sess_config = tf.ConfigProto(
                 allow_soft_placement=True,
                 log_device_placement=False)
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def load_data(train_filename, valid_filename, test_filename, delimiter='\t', col_names=['user_id', 'item_id', 'rating']):
    """Helper function to load in/preprocess dataframes"""
    train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
    train_data['user_id'] = train_data['user_id'] - 1
    train_data['item_id'] = train_data['item_id'] - 1
    valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
    valid_data['user_id'] = valid_data['user_id'] - 1
    valid_data['item_id'] = valid_data['item_id'] - 1
    test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
    test_data['user_id'] = test_data['user_id'] - 1
    test_data['item_id'] = test_data['item_id'] - 1

    return train_data, valid_data, test_data

def train(model, sess, saver, train_data, valid_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch):
    # Print initial values
    batch = train_data.sample(batch_size) if batch_size else train_data
    train_error = model.eval_loss(batch)
    train_rmse = model.eval_rmse(batch)
    valid_rmse = model.eval_rmse(valid_data)
    print("[start] Train error: {:3f}, Train RMSE: {:3f}; Valid RMSE: {:3f}".format(train_error, train_rmse, valid_rmse))

    # Optimize
    prev_valid_rmse = float("Inf")
    early_stop_epochs = 0
    #for epoch in xrange(max_epochs):
    for epoch in range(max_epochs):
        # Run (S)GD
        shuffled_df = train_data.sample(frac=1)
        batches = chunk_df(shuffled_df, batch_size) if batch_size else [train_data]

        for batch_iter, batch in enumerate(batches):
            model.train_iteration(batch)

            # Evaluate
            train_error = model.eval_loss(batch)
            train_rmse = model.eval_rmse(batch)
            valid_rmse = model.eval_rmse(valid_data)
            print("[{:d}-{:d}] Train error: {:3f}, Train RMSE: {:3f}; Valid RMSE: {:3f}".format(epoch, batch_iter, train_error, train_rmse, valid_rmse))

        # Checkpointing/early stopping
        if use_early_stop:
            early_stop_epochs += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_epochs = 0
                saver.save(sess, model.model_filename)
            elif early_stop_epochs == early_stop_max_epoch:
                print("Early stopping ({} vs. {})...".format(prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(sess, model.model_filename)

def test(model, sess, saver, test_data, train_data=None, log=True):
    if train_data is not None:
        train_rmse = model.eval_rmse(train_data)
        if log:
            print("Final train RMSE: {}".format(train_rmse))

    test_rmse = model.eval_rmse(test_data)
    if log:
        print("Final test RMSE: {}".format(test_rmse))

    return test_rmse

if __name__ == '__main__':
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates NNMF models.')
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['NNMF', 'SVINNMF'],
                        help='the name of the model to use', required=True)
    parser.add_argument('--mode', metavar='MODE', type=str, choices=['train', 'select', 'test'],
                        help='the mode to run the program in', required=True)
    parser.add_argument('--train', metavar='TRAIN_INPUT_FILE', type=str, default='data/ml-1m/split/ratings.dat.train',
                        help='the location of the training set\'s input file')
    parser.add_argument('--valid', metavar='VALID_INPUT_FILE', type=str, default='data/ml-1m/split/ratings.dat.valid',
                        help='the location of the validation set\'s input file')
    parser.add_argument('--test', metavar='TEST_INPUT_FILE', type=str, default='data/ml-1m/split/ratings.dat.test',
                        help='the location of the test set\'s input file')
    parser.add_argument('--users', metavar='NUM_USERS', type=int, default=6040, # ML 100K has 943 users
                        help='the number of users in the data set')
    parser.add_argument('--movies', metavar='NUM_MOVIES', type=int, default=3952, # ML 100K has 1682 movies
                        help='the number of movies in the data set')
    parser.add_argument('--model-params', metavar='MODEL_PARAMS_JSON', type=str, default='{}',
                        help='JSON string containing model params')
    parser.add_argument('--delim', metavar='DELIMITER', type=str, default='\t',
                        help='the delimiter to use when parsing input files')
    parser.add_argument('--cols', metavar='COL_NAMES', type=str, default=['user_id', 'item_id', 'rating'],
                        help='the column names of the input data', nargs='+')
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=25000,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--no-early', default=False, action='store_true',
                        help='disable early stopping')
    parser.add_argument('--early-stop-max-epoch', metavar='EARLY_STOP_MAX_EPOCH', type=int, default=40,
                        help='the maximum number of epochs to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-epochs', metavar='MAX_EPOCHS', type=int, default=1000,
                        help='the maximum number of epochs to allow the model to train for')
    parser.add_argument('--hyperparam-search-size', metavar='HYPERPARAM_SEARCH_SIZE', type=int, default=50,
                        help='when in "select" mode, the number of times to sample for random search')

    # Parse args
    args = parser.parse_args()
    # Global args
    model_name = args.model
    mode = args.mode
    train_filename = args.train
    valid_filename = args.valid
    test_filename = args.test
    num_users = args.users
    num_items = args.movies
    model_params = json.loads(args.model_params)
    delimiter = args.delim
    col_names = args.cols
    batch_size = args.batch
    use_early_stop = not(args.no_early)
    early_stop_max_epoch = args.early_stop_max_epoch
    max_epochs = args.max_epochs

    if mode in ('train', 'test'):
        with tf.Session(config=sess_config) as sess:
            # Define computation graph & Initialize
            print('Building network & initializing variables')
            if model_name == 'NNMF':
                model = NNMF(num_users, num_items, **model_params)
            elif model_name == 'SVINNMF':
                model = SVINNMF(num_users, num_items, **model_params)
            else:
                raise NotImplementedError("Model '{}' not implemented".format(model_name))

            model.init_sess(sess)
            saver = tf.train.Saver()

            # Train
            if mode in ('train', 'test'):
                # Process data
                print("Reading in data")
                train_data, valid_data, test_data = load_data(train_filename, valid_filename, test_filename,
                    delimiter=delimiter, col_names=col_names)

                if mode == 'train':
                    # Create model directory, if needed
                    if not os.path.exists(os.path.dirname(model.model_filename)):
                        os.makedirs(os.path.dirname(model.model_filename))

                    # Train
                    train(model, sess, saver, train_data, valid_data, batch_size=batch_size, max_epochs=max_epochs,
                          use_early_stop=use_early_stop, early_stop_max_epoch=early_stop_max_epoch)

                print('Loading best checkpointed model')
                saver.restore(sess, model.model_filename)
                test(model, sess, saver, test_data, train_data=train_data)

    elif mode == 'select':
        print('Training model with multiple hyperparameter variations')

        hyperparam_search_size = args.hyperparam_search_size

        # Generate list of hyperparams (model param dicts)
        hyperparams_list = []
        best_setting, best_rmse, best_test_rmse = None, float('Inf'), None

        _NNMF_LAM_MIN, _NNMF_LAM_MAX = -4.0, 4.0
        _SVINNMF_VAR_MIN, _SVINNMF_VAR_MAX = -2.0, 2.0
        _SVINNMF_KL_EPOCH_MIN, _SVINNMF_KL_EPOCH_MAX = 2.0, 4.0
        #for _ in xrange(hyperparam_search_size):
        for _ in range(hyperparam_search_size):
            if model_name == 'NNMF':
                hyperparams_list.append({'lam': 10 ** np.random.uniform(_NNMF_LAM_MIN, _NNMF_LAM_MAX)})
            elif model_name == 'SVINNMF':
                hyperparams_list.append({
                    'U_prior_var': 10 ** np.random.uniform(_SVINNMF_VAR_MIN, _SVINNMF_VAR_MAX),
                    'Uprime_prior_var': 10 ** np.random.uniform(_SVINNMF_VAR_MIN, _SVINNMF_VAR_MAX),
                    'V_prior_var': 10 ** np.random.uniform(_SVINNMF_VAR_MIN, _SVINNMF_VAR_MAX),
                    'Vprime_prior_var': 10 ** np.random.uniform(_SVINNMF_VAR_MIN, _SVINNMF_VAR_MAX),
                    'kl_full_epoch': int(10 ** np.random.uniform(_SVINNMF_KL_EPOCH_MIN, _SVINNMF_KL_EPOCH_MAX))
                })

        # Setup folder to store models
        timestamp = int(time.time())
        select_dir = "select/{}/{}".format(model_name, timestamp)
        if not os.path.exists(select_dir):
            os.makedirs(select_dir)

        # Run for each setting
        print("Running for hyperparam settings: {}".format(hyperparams_list))
        for idx, hyperparams in enumerate(hyperparams_list):
            print("----\n{}: {}\n----".format(idx, hyperparams))

            # Update model params with hyperparams and model filename
            model_params.update(hyperparams)
            model_params.update({'model_filename': os.path.join(select_dir, "{}.ckpt".format(str(idx)))})

            with tf.Session(config=sess_config) as sess:
                # Define computation graph & Initialize
                print('Building network & initializing variables')
                if model_name == 'NNMF':
                    model = NNMF(num_users, num_items, **model_params)
                elif model_name == 'SVINNMF':
                    model_params.update({'kl_full_epoch': 200})
                    model = SVINNMF(num_users, num_items, **model_params)
                else:
                    raise NotImplementedError("Model '{}' not implemented".format(model_name))

                model.init_sess(sess)
                saver = tf.train.Saver()

                # Train
                print("Reading in data")
                train_data, valid_data, test_data = load_data(train_filename, valid_filename, test_filename,
                    delimiter=delimiter, col_names=col_names)

                train(model, sess, saver, train_data, valid_data, batch_size=batch_size, max_epochs=max_epochs,
                      use_early_stop=use_early_stop, early_stop_max_epoch=early_stop_max_epoch)

                print('Loading best checkpointed model')
                saver.restore(sess, model.model_filename)
                valid_rmse = model.eval_rmse(valid_data)

                # Log the results to file
                with open(os.path.join(select_dir, 'results.tsv'), 'a') as log_out:
                    log_out.write("{}\t{:3f}\t{}\n".format(idx, valid_rmse, hyperparams))

                # Update the best setting, if applicable
                if valid_rmse < best_rmse:
                    best_rmse = valid_rmse
                    best_setting = hyperparams
                    best_test_rmse = test(model, sess, saver, test_data, log=False)

        # Spit out the best hyperparams and the model's performance on test set
        print("\n====\nBest setting: {} ({:3f})".format(best_setting, best_rmse))
        print("Final test RMSE: {:3f}".format(best_test_rmse))
        print('====')

    else:
        raise Exception("Mode '{}' not available".format(mode))
