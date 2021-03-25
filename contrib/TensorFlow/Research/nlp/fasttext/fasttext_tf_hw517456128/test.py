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

import os
import json
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1.logging as logging
import tensorflow as tf

from fasttext_utils import (
    parse_txt,
    next_batch,
    get_all,
)
from utils import load_graph

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

#logging.set_verbosity(logging.ERROR)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--model_dir", type=str, help="path where model.pb and model_params.json are")
    parser.add_argument("-tp", "--test_path", type=str, help="path to test file")
    parser.add_argument("-lp", "--label_prefix", type=str, help="label prefix", default="__label__")
    parser.add_argument("-bs", "--batch_size", type=int, default=1024, help="batch size for inference")
    parser.add_argument("-k", "--top_k", type=int, default=1, help="calculate accuracy on top k predictions")
    parser.add_argument("-hc", "--hand_check", type=bool, default=False, help="test on manually inputted data")
    parser.add_argument("-gpu", "--use_gpu", type=bool, default=True, help="use gpu for inference")
    parser.add_argument("-gpu_fr", "--gpu_fraction", type=float, default=0.4, help="what fraction of gpu to allocate")
    args = parser.parse_args()

    model_dir = args.model_dir
    model_params_path = os.path.join(model_dir, "model_params.json")
    model_path = os.path.join(model_dir, "model_best.pb")
    label_prefix = args.label_prefix

    # if args.use_gpu:
    #     device = "/gpu:0"
    #     config = tf.ConfigProto(allow_soft_placement=True,
    #                             gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
    #                                                       allow_growth=True))
    # else:
    #     device = "/cpu:0"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #     config = tf.ConfigProto(allow_soft_placement=True)

    num_thrown_for_label = 0
    with open(model_params_path, "r") as infile:
        model_params = json.load(infile)
    if os.path.isfile(model_params["label_dict_path"]):
        with open(model_params["label_dict_path"], "r") as infile:
            label_dict = json.load(infile)
    else:
        with open(os.path.join(model_dir, "label_dict.json"), "r") as infile:
            label_dict = json.load(infile)
    if os.path.isfile(model_params["word_dict_path"]):
        with open(model_params["word_dict_path"], "r") as infile:
            word_dict = json.load(infile)
    else:
        with open(os.path.join(model_dir, "word_dict.json"), "r") as infile:
            word_dict = json.load(infile)
    word_ngrams = model_params["word_ngrams"]
    sort_ngrams = model_params["sort_ngrams"]

    labels_dict_inverse = {}

    for label, label_id in label_dict.items():
        labels_dict_inverse[label_dict[label]["id"]] = label

    with tf.Session(config=config) as sess:
        run_arg = load_graph(model_path, ["input:0", "input_weights:0", "prediction:0"])
        if args.hand_check:
            while True:
                query_description = input("Enter the description: ")
                label = [token for token in query_description.split() if token.startswith(label_prefix)][0]
                label = label.split(label_prefix)[-1]
                query_description = query_description[20:]
                test_description_indices = \
                    np.expand_dims([0] + [word_dict[phrase]["id"] for phrase in
                                            get_all(query_description.split(), word_ngrams, sort_ngrams)
                                            if phrase in word_dict], axis=0)

                test_desc_weights = np.zeros_like(test_description_indices, dtype=np.float32)
                test_desc_weights[0][:len(test_description_indices[0])] = 1. / len(test_description_indices[0])

                if label not in label_dict:
                    print("New label")
                    continue

                probabilities = np.squeeze(sess.run(run_arg[-1], feed_dict={run_arg[0]: test_description_indices,
                                                                                run_arg[1]: test_desc_weights}))

                max_index = np.argmax(probabilities)
                max_prob = probabilities[max_index]
                predicted_label = labels_dict_inverse[max_index]
                print(predicted_label == label, predicted_label, max_prob)
        else:
            test_descriptions, test_labels = parse_txt(args.test_path)
            test_indices = np.arange(len(test_descriptions))
            print("The total number of test datapoints: {}".format(len(test_descriptions)))

            progress_bar = tqdm(total=int(np.ceil(len(test_descriptions) / args.batch_size)))
            remaining_indices, batch_indices = next_batch(test_indices, args.batch_size)
            accuracy_top_1, accuracy_top_k = 0, 0
            cnt = 0

            while len(batch_indices) > 0:
                batch_descriptions = [test_descriptions[i] for i in batch_indices]
                batch_labels = [test_labels[i] for i in batch_indices]

                batch, batch_weights, batch_labels2 = [], [], []

                max_words = -1
                for test_description in batch_descriptions:
                    max_words = max(max_words, len(test_description.split()))

                num_max_words = 1
                for ng in range(word_ngrams):
                    num_max_words += max_words - ng

                for test_description, test_label in zip(batch_descriptions, batch_labels):
                    if test_label not in label_dict:
                        num_thrown_for_label += 1
                        continue
                    initial_test_indices = [0] + [word_dict[phrase]["id"] for phrase in
                                                    get_all(test_description.split(), word_ngrams, sort_ngrams)
                                                    if phrase in word_dict]

                    cnt += 1
                    test_description_indices = \
                        np.array(initial_test_indices +
                                    [0 for _ in range(num_max_words - len(initial_test_indices))])
                    test_description_weights = np.zeros_like(test_description_indices, dtype=np.float32)
                    test_description_weights[:len(initial_test_indices)] = 1. / len(initial_test_indices)

                    batch.append(test_description_indices)
                    batch_weights.append(test_description_weights)
                    batch_labels2.append(label_dict[test_label]["id"])

                probabilities = sess.run(run_arg[-1], feed_dict={run_arg[0]: batch, run_arg[1]: batch_weights})
                top_k = [np.argsort(i)[-args.top_k:] for i in probabilities]

                accuracy_top_k += sum([True if i in j else False for i, j in zip(batch_labels2, top_k)])
                accuracy_top_1 += sum([True if i == j[-1] else False for i, j in zip(batch_labels2, top_k)])
                remaining_indices, batch_indices = next_batch(remaining_indices, args.batch_size)
                progress_bar.update()
            progress_bar.close()

            print("{} datapoint thrown because of label".format(num_thrown_for_label))
            print("Number of test datapoints after cleaning: {}".format(len(test_descriptions) -
                                                                            num_thrown_for_label))
            print("Number of unique labels in test after cleaning: {}".format(len(set(test_labels))))
            print("Accuracy: {}".format(round(100 * accuracy_top_1 / len(test_descriptions), 2)))
            print("Accuracy top {}: {}".format(args.top_k, round(100 * accuracy_top_k / len(test_descriptions), 2)))


if __name__ == "__main__":
    main()
