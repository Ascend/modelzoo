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

import json
import time
import argparse
import tracemalloc
from sys import exit
import moxing as mox
import os
import tensorflow as tf

import numpy as np
import npu_bridge
import tensorflow.compat.v1.logging as logging
#from npu_bridge.estimator.npu.npu_config import NPURunConfig
#from npu_bridge.estimator.npu.npu_config import DumpConfig


from utils import (
    hash_,
    validate,
    get_cache_hash,
)
from train import (
    run_train,
    get_accuracy,
)
from fasttext_utils import (
    parse_txt,
    clean_directory,
    cache_data,
    check_model_presence,
    get_word_label_vocabs,
    get_max_words_with_ngrams,
)

#logging.set_verbosity(logging.ERROR)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ['SLOG_PRINT_TO_STDOUT'] = '1' # remove
#os.environ['DUMP_GE_GRAPH'] = '2'

#TMP_CACHE_DIR = '/cache/data'
#os.makedirs(TMP_CACHE_DIR)
#dump_config = DumpConfig(enable_dump=True, dump_path=TMP_CACHE_DIR, dump_step="0") #, dump_debug="all")


def main():
    main_start = time.time()
    tracemalloc.start()
    parser = argparse.ArgumentParser()

    # data specific parameters
    parser.add_argument("-du", "--data_url", type=str, help="path to train file", default="")
    parser.add_argument("-tru", "--train_url", type=str, help="path to train file", default="")
    parser.add_argument("-trp", "--train_file", type=str, help="path to train file", default="/train.txt")
    parser.add_argument("-tp", "--test_file", type=str, help="path to test file", default="/test.txt")
    parser.add_argument("-lp", "--label_prefix", type=str, help="label prefix", default="__label__")
    parser.add_argument("-df", "--data_fraction", type=float, default=1, help="data fraction")
    parser.add_argument("-seed", "--seed", type=int, default=17)

    # hyper-parameters
    parser.add_argument("-dim", "--embedding_dim", type=int, default=10, help="length of embedding vector")
    parser.add_argument("-nep", "--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-wng", "--word_ngrams", type=int, default=1, help="word ngrams")
    parser.add_argument("-sng", "--sort_ngrams", type=int, default=0, help="sort n-grams alphabetically")
    parser.add_argument("-bs", "--batch_size", type=int, default=4096, help="batch size for train")
    parser.add_argument("-bn", "--use_batch_norm", type=int, default=0, help="use batch norm")
    parser.add_argument("-mwc", "--min_word_count", type=int, default=1,
                        help="discard words which appear less than this number")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.3, help="learning rate")
    parser.add_argument("-lrm", "--learning_rate_multiplier", type=float, default=0.8, help="learning rate multiplier")
    parser.add_argument("-dr", "--dropout", type=float, default=0.5, help="train dropout keep rate")
    parser.add_argument("-l2", "--l2_reg_weight", type=float, default=1e-6, help="regularization weight")

    # parameters
    parser.add_argument("-bsi", "--batch_size_inference", type=int, default=4096, help="batch size for test")
    parser.add_argument("-k", "--top_k", type=int, default=3, help="report results for top k predictions")
    parser.add_argument("-ck", "--compare_top_k", type=int, default=0,
                        help="compare top k accuracies for determining the best model")
    parser.add_argument("-sm", "--save_all_models", type=int, default=0, help="save model after each epoch")
    parser.add_argument("-ut", "--use_test", type=int, default=0, help="evaluate on test data")
    parser.add_argument("-gpu", "--use_gpu", type=int, default=0, help="use gpu for training")
    parser.add_argument("-gpu_fr", "--gpu_fraction", type=float, default=0.5, help="what fraction of gpu to allocate")
    parser.add_argument("-utb", "--use_tensorboard", type=int, default=0, help="use tensorboard")
    parser.add_argument("-cd", "--cache_dir", type=str, help="cache directory", default="/cache/")
    parser.add_argument("-ld", "--log_dir", type=str, help="log directory", default="/results/")
    parser.add_argument("-f", "--force", type=int, default=0, help="force retraining")
    parser.add_argument("-pb", "--progress_bar", type=int, default=1, help="show progress bar")
    parser.add_argument("-fl", "--flush", type=int, default=0, help="flush after print")

    args = parser.parse_args()
    for bool_param in [args.use_batch_norm, args.save_all_models, args.use_test, args.sort_ngrams, args.use_gpu,
                       args.use_tensorboard, args.force, args.flush, args.compare_top_k, args.progress_bar]:
        if bool_param not in [0, 1]:
            raise ValueError("{} should be 0 or 1.".format(bool_param))

    #npu_config = NPURunConfig(dump_config=dump_config, model_dir=args.train_url,
    #                          session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    '''
    temp_dir = '/cache'
    local_dir = os.path.join(temp_dir, "dataset-fasttext")
    if os.path.isdir(local_dir):
        print("===>>>Directory:{} exist...".format(local_dir))
    else:
        print("===>>>Directory:{} not exist. generate it!".format(local_dir))
        os.makedirs(local_dir)
        import datetime
        start = datetime.datetime.now()
        #mox.file.copy_parallel("./", args.train_url)
        #mox.file.copy_parallel(TMP_CACHE_DIR, args.train_url)
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_dir)
        end = datetime.datetime.now()
        print("===>>>Copy from obs to local, time use:{}(s)".format((end - start).seconds))
        print("===>>>Copy files from obs:{} to local dir:{}".format(args.data_url, local_dir))
    '''
    local_dir = args.data_url

    train_path = os.path.abspath(local_dir+args.train_file)
    sort_ngrams = bool(args.sort_ngrams)
    progress_bar = bool(args.progress_bar)
    flush = bool(args.flush)

    use_test = False
    if args.test_file:
        args.test_path = os.path.abspath(local_dir+args.test_file)
        if bool(args.use_test):
            use_test = True

    print("\n\nTraining with arguments:\n{}\n".format(args))

    cache_dir = validate(local_dir+args.cache_dir)
    log_dir = validate(local_dir+args.log_dir)
    train_history_path = os.path.join(log_dir, "history.json")

    np.random.seed(args.seed)

    train_descriptions, train_labels, max_words = \
        parse_txt(train_path, as_tokens=True, return_max_len=True,
                  fraction=args.data_fraction, seed=args.seed, label_prefix=args.label_prefix)

    data_specific = {
        "seed": args.seed, "data_fraction": args.data_fraction, "min_word_count": args.min_word_count,
        "word_ngrams": args.word_ngrams, "sort_ngrams": sort_ngrams,
    }

    data_hash = get_cache_hash(list_of_texts=train_descriptions, data_specific_params=data_specific)
    cache_dir = os.path.abspath(validate(os.path.join(cache_dir, data_hash)))

    train_specific = {"embedding_dim": args.embedding_dim, "num_epochs": args.num_epochs, "batch_size": args.batch_size,
                      "learning_rate": args.learning_rate, "learning_rate_multiplier": args.learning_rate_multiplier,
                      "use_batch_norm": bool(args.use_batch_norm), "l2_reg_weight": args.l2_reg_weight,
                      "dropout": args.dropout, "cache_dir": cache_dir}

    for k, v in data_specific.items():
        train_specific[k] = v

    model_params = {
        "word_ngrams": args.word_ngrams,
        "sort_ngrams": sort_ngrams,
        "word_dict_path": os.path.abspath(os.path.join(cache_dir, "word_dict.json")),
        "label_dict_path": os.path.abspath(os.path.join(cache_dir, "label_dict.json"))
    }

    hyperparams_hashed = hash_("".join([str(i) for i in train_specific.values()]))
    current_log_dir = validate(os.path.join(log_dir, hyperparams_hashed))
    data_specific["train_path"], train_specific["train_path"] = train_path, train_path

    train_params = {
        "use_gpu": bool(args.use_gpu),
        "gpu_fraction": args.gpu_fraction,
        "use_tensorboard": bool(args.use_tensorboard),
        "top_k": args.top_k,
        "save_all_models": bool(args.save_all_models),
        "compare_top_k": bool(args.compare_top_k),
        "use_test": use_test,
        "log_dir": current_log_dir,
        "batch_size_inference": args.batch_size_inference,
        "progress_bar": progress_bar,
        "flush": flush,
    }

    if os.path.exists(train_history_path):
        with open(train_history_path) as infile:
            train_history = json.load(infile)

        if hyperparams_hashed in train_history and check_model_presence(current_log_dir):
            if not bool(args.force):
                if args.test_path:
                    get_accuracy(current_log_dir, train_params, train_history_path, hyperparams_hashed,
                                 train_history, args.test_path, args.label_prefix)
                else:
                    get_accuracy(current_log_dir, train_params, train_history_path, hyperparams_hashed,
                                 train_history, train_path, args.label_prefix)

                print("The model is stored at {}".format(current_log_dir))
                exit()
            else:
                print("Forced retraining")
                print("Training hyper-parameters hashed: {}".format(hyperparams_hashed))
        else:
            print("Training hyper-parameters hashed: {}".format(hyperparams_hashed))
    else:
        train_history = dict()

    clean_directory(current_log_dir)

    max_words_with_ng = get_max_words_with_ngrams(max_words, args.word_ngrams)

    print("Preparing dataset")
    print("Total number of datapoints: {}".format(len(train_descriptions)))
    print("Max number of words in description: {}".format(max_words))
    print("Max number of words with n-grams in description: {}".format(max_words_with_ng))

    word_vocab, label_vocab = get_word_label_vocabs(train_descriptions, train_labels, args.word_ngrams,
                                                    args.min_word_count, sort_ngrams, cache_dir, bool(args.force),
                                                    show_progress=progress_bar, flush=flush)

    with open(os.path.join(current_log_dir, "model_params.json"), "w+") as outfile:
        json.dump(model_params, outfile)

    num_words_in_train = len(word_vocab)
    train_description_hashes, train_labels, cache = \
        cache_data(train_descriptions, train_labels, word_vocab, label_vocab, args.word_ngrams, sort_ngrams,
                   show_progress=progress_bar, progress_desc="Cache train descriptions", flush=flush)
    del train_descriptions

    test_description_hashes, test_labels = [], []
    initial_test_len = 0
    if use_test:
        test_descriptions, test_labels, max_words_test = parse_txt(args.test_path, as_tokens=True, return_max_len=True,
                                                                   label_prefix=args.label_prefix)
        initial_test_len = len(test_descriptions)

        print("Total number of test datapoints: {}".format(len(test_descriptions)))
        test_description_hashes, test_labels, cache = \
            cache_data(test_descriptions, test_labels, word_vocab, label_vocab, args.word_ngrams, sort_ngrams,
                       cache=cache, is_test_data=True, show_progress=progress_bar,
                       progress_desc="Cache test descriptions", flush=flush)
        del test_descriptions

    data = {
        "train_description_hashes": train_description_hashes,
        "train_labels": train_labels,
        "test_description_hashes": test_description_hashes,
        "test_labels": test_labels,
        "cache": cache,
        "label_vocab": label_vocab,
        "num_words_in_train": num_words_in_train,
        "test_path": args.test_path,
        "initial_test_len": initial_test_len,
    }

    run_train(data, train_specific, train_params, data_specific, train_history, train_history_path)
    print("All process took {} seconds".format(round(time.time() - main_start, 0)), flush=flush)


if __name__ == "__main__":
    main()
