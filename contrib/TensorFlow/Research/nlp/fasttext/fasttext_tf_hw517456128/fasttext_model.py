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

import inspect
import json
import os
import warnings
from subprocess import (
    Popen,
    PIPE,
    STDOUT,
)

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow.compat.v1.logging as logging

from utils import (
    load_graph,
    hash_,
    validate,
    handle_space_paths,
    copy_all,
)
from fasttext_utils import (
    get_all,
    parse_txt,
    next_batch,
    preprocess_data,
    get_accuracy_log_dir,
)

#logging.set_verbosity(logging.ERROR)
#warnings.filterwarnings("ignore")

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FastTextModel(object):
    def __init__(self, model_path, model_params_path, label_prefix="__label__", preprocessing_function=None,
                 use_gpu=True, gpu_fraction=0.5, hyperparams=None):
        """
        :param model_path: str, path to pb file
        :param model_params_path: str, path to pb model_params.json
        :param label_prefix: list, prefix for labels
        :param preprocessing_function: function, function to apply on data
        :param use_gpu: bool, use gpu for training
        :param gpu_fraction: float, gpu fraction to allocate
        :param hyperparams: dict, all hyperparams for train_supervised
        :return: object, the trained model
        """
        tf.reset_default_graph()
        self._graph = tf.Graph()
        self.label_prefix = label_prefix
        if hyperparams:
            self.hyperparams = hyperparams
        else:
            self.hyperparams = dict()
        self.info = {"model_path": os.path.abspath(model_path), "model_params_path": os.path.abspath(model_params_path)}
        with open(model_params_path, "r") as infile:
            model_params = json.load(infile)
        for key, value in model_params.items():
            self.info[key] = value
        if os.path.isfile(model_params["label_dict_path"]):
            with open(model_params["label_dict_path"], "r") as infile:
                self.label_dict = json.load(infile)
        else:
            new_path = os.path.join(os.path.dirname(model_params_path), "label_dict.json")
            print("{} not found, switching to model_params' path {}".format(model_params["label_dict_path"], new_path))
            with open(new_path, "r") as infile:
                self.label_dict = json.load(infile)
            self.info["label_dict_path"] = os.path.abspath(new_path)
        if os.path.isfile(model_params["word_dict_path"]):
            with open(model_params["word_dict_path"], "r") as infile:
                self.word_dict = json.load(infile)
        else:
            new_path = os.path.join(os.path.dirname(model_params_path), "word_dict.json")
            print("{} not found, switching to model_params' path {}".format(model_params["word_dict_path"], new_path))
            with open(new_path, "r") as infile:
                self.word_dict = json.load(infile)
            self.info["word_dict_path"] = os.path.abspath(new_path)
        self.preprocessing_function = preprocessing_function

        get_list = ["input", "input_weights", "embeddings/embedding_matrix/read",
                    "mean_sentence_embedding/sentence_embedding", "logits/kernel/read", "prediction"]
        get_list = [i + ":0" for i in get_list]

        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        #self._device = "/cpu:0"
        #config = tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
        #if use_gpu:
        #    self._device = "/gpu:0"
        #    config = tf.ConfigProto(allow_soft_placement=True,
        #                            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
        #                                                      allow_growth=True))
        self._input_matrix, self._output_matrix = None, None

        #with tf.device(self._device):
        with self._graph.as_default():
            self._input_placeholder, self._weights_placeholder, self._input_matrix_tensor, self._sentence_vector, \
                self._output_matrix_tensor, self._output = load_graph(model_path, get_list)

        self._sess = tf.Session(graph=self._graph, config=config)
        self._dim = self.get_dimension()
        _ = self.predict([""] * 3, batch_size=3, show_progress=False)  # warm up

    def __del__(self):
        #with tf.device(self._device):
        self._sess.close()

    def get_dimension(self):
        """
        Get the dimension (size) of a lookup vector (hidden layer).
        :return: int
        """
        return int(self._sentence_vector.shape[1])

    def get_input_matrix(self):
        """
        Get a copy of the full input matrix of a Model.
        :return: np.ndarray, size: word_count * dim
        """
        if self._input_matrix is None:
            self._input_matrix = self._sess.run(self._input_matrix_tensor)
        return self._input_matrix

    def get_input_vector(self, index):
        """
        Given an index, get the corresponding vector of the Input Matrix.
        :param index: int
        :return: np.ndarray, size: dim
        """
        return self._input_matrix[index]

    def get_labels(self, include_freq=False):
        """
        Get the entire list of labels of the dictionary optionally including the frequency of the individual labels.
        :param include_freq: bool, returns tuple with labels and their frequencies
        :return: list / tuple of lists
        """
        labels = sorted(self.label_dict.keys())
        if include_freq:
            return labels, [self.label_dict[key]["cnt"] for key in labels]
        return labels

    def get_line(self, text):
        """
        Preprocess the text and split it into words and labels. Labels must start with the prefix used to create the
        model (__label__ by default) and have been used in training.
        :param text: str
        :return: (list, list)
        """
        tokens, labels = ["__MEAN_EMBEDDING__"], []
        for token in text.split():
            if token.startswith(self.label_prefix):
                label_clean = token[len(self.label_prefix):]
                if label_clean in self.label_dict:
                    labels.append(label_clean)
            else:
                tokens.append(token)
        if self.preprocessing_function:
            tokens = self.preprocessing_function(" ".join(tokens)).split()
        return tokens, labels

    def get_output_matrix(self):
        """
        Get a copy of the full output matrix of a Model.
        :return: np.ndarray, size: dim * label_count
        """
        if self._output_matrix is None:
            self._output_matrix = self._sess.run(self._output_matrix_tensor)
        return self._output_matrix

    def get_sentence_vector(self, text, batch_size=1000):
        """
        Given a string or list of string, get its (theirs) vector represenation(s). This function applies
        preprocessing function on the strings.
        :param text: str or list/array
        :param batch_size: int
        :return: np.ndarray, size: dim
        """

        if not isinstance(text, (list, str, np.ndarray, pd.Series)):
            raise ValueError("text should be string, list, numpy array or pandas series")
        if isinstance(text, str):
            text = [text]
        embeddings = []

        for batch, batch_weights in self._batch_generator(text, batch_size):
            embeddings.extend(self._sess.run(self._sentence_vector,
                                             feed_dict={self._input_placeholder: batch,
                                                        self._weights_placeholder: batch_weights}))
        return np.squeeze(embeddings)

    def get_subword_id(self, subword):
        """
        Given a subword, get the word id within the dictionary. Returns -1 if word is not in the dictionary.
        :param subword:
        :return: int. Returns -1 if is not in vocabulary
        """
        return self.word_dict[subword]["id"] if subword in self.word_dict else -1

    def get_subwords(self, word):
        word = word.replace("_", " ")
        word_splitted = word.split()
        if len(word_splitted) > self.info["word_ngrams"]:
            return [], []
        else:
            subwords = [phrase for phrase in get_all(word_splitted, self.info["word_ngrams"], self.info["sort_ngrams"])
                        if phrase in self.word_dict]
            return subwords, [self.get_word_id(subword) for subword in subwords]

    def get_word_id(self, word):
        if " " in word:
            word = word.replace(" ", "_")
        return self.word_dict[word]["id"] if word in self.word_dict else -1

    def get_word_vector(self, word):
        """
        Get the vector representation of word.
        :param word: str
        :return: np.ndarray, size: dim. returns 0s if not from vocabulary
        """
        if self.preprocessing_function:
            word_dict = self.get_word_id(self.preprocessing_function(word))
        else:
            word_dict = self.get_word_id(word)
        return self.get_input_vector(word_dict) if word_dict != -1 else np.zeros(self._dim, dtype=np.float32)

    def get_words(self, include_freq=False):
        """
        Get the entire list of words of the dictionary optionally including the frequency of the individual words.
        :param include_freq: bool, returns tuple with words and their frequencies
        :return: list / tuple of lists
        """
        words = sorted(self.word_dict.keys())
        if include_freq:
            return words, [self.word_dict[key]["cnt"] for key in words]
        return words

    def _batch_generator(self, list_of_texts, batch_size, show_progress=False):
        """
        Generate batch from list of texts
        :param list_of_texts: list/array
        :param batch_size: int
        :param show_progress: bool, show progress bar
        :return: batch word indices, batch word weights
        """
        if self.preprocessing_function:
            list_of_texts = [self.preprocessing_function(str(text)) for text in list_of_texts]
        else:
            list_of_texts = [str(text) for text in list_of_texts]
        indices = np.arange(len(list_of_texts))
        remaining_indices, batch_indices = next_batch(indices, batch_size)

        if len(list_of_texts) <= batch_size:
            show_progress = False

        disable_progress_bar = not show_progress
        progress_bar = tqdm(total=int(np.ceil(len(list_of_texts) / batch_size)), disable=disable_progress_bar)

        while len(batch_indices) > 0:
            batch, batch_weights = [], []

            batch_descriptions = [list(get_all(list_of_texts[index].split(), self.info["word_ngrams"],
                                               self.info["sort_ngrams"])) for index in batch_indices]
            num_max_words = max([len(batch_description) for batch_description in batch_descriptions]) + 1

            for batch_description in batch_descriptions:
                initial_indices = [0] + [self.word_dict[phrase]["id"] for phrase in batch_description
                                         if phrase in self.word_dict]

                description_indices = np.array(initial_indices +
                                               [0 for _ in range(num_max_words - len(initial_indices))])
                description_weights = np.zeros_like(description_indices, dtype=np.float32)
                description_weights[:len(initial_indices)] = 1. / len(initial_indices)

                batch.append(description_indices)
                batch_weights.append(description_weights)
            remaining_indices, batch_indices = next_batch(remaining_indices, batch_size)

            progress_bar.update()
            yield batch, batch_weights

        progress_bar.close()

    def predict(self, list_of_texts, k=1, threshold=-0.1, batch_size=1000, show_progress=True):
        """
        Predict top k predictions on given texts
        :param list_of_texts: list/array
        :param k: int, top k predictions
        :param threshold: float, from 0 to 1, default -0.1 meaining no threshold
        :param batch_size: int
        :param show_progress: bool, ignored if list of text is string or has smaller or equal length to batch size
        :return: top k predictions and probabilities
        """
        if isinstance(list_of_texts, str):
            list_of_texts = [list_of_texts]

        labels = self.get_labels()
        predictions, probabilities = [], []

        for batch, batch_weights in self._batch_generator(list_of_texts, batch_size, show_progress):
            batch_probabilities = self._sess.run(self._output, feed_dict={self._input_placeholder: batch,
                                                                          self._weights_placeholder: batch_weights})

            top_k_probabilities, top_k_predictions = [], []
            for i in batch_probabilities:
                predictions_row, probabilities_row = [], []
                if k == -1:
                    top_k_indices = np.argsort(i)[::-1]
                else:
                    top_k_indices = np.argsort(i)[-k:][::-1]
                for index, probability in zip(top_k_indices, i[top_k_indices]):
                    if probability > threshold:
                        predictions_row.append(index)
                        probabilities_row.append(probability)
                top_k_predictions.append([labels[i] for i in predictions_row])
                top_k_probabilities.append(probabilities_row)
            predictions.extend(top_k_predictions)
            probabilities.extend(top_k_probabilities)
        return predictions, probabilities

    def test(self, list_of_texts, list_of_labels, k=1, threshold=-0.1, batch_size=1000, show_progress=True):
        """
        Predict top k predictions on given texts
        :param list_of_texts: list/array
        :param list_of_labels: list/array
        :param k: int, top k predictions
        :param threshold: float, from 0 to 1. Default is -0.1 meaining no threshold
        :param batch_size: int
        :param show_progress: bool
        :return: top k predictions and probabilities
        """
        if len(list_of_texts) != len(list_of_labels):
            raise ValueError('the lengths of list_of_texts and list_of_labels must match')

        predictions, probabilities = self.predict(list_of_texts=list_of_texts, batch_size=batch_size, k=k,
                                                  threshold=threshold, show_progress=show_progress)
        recall, precision = 0, 0
        all_labels, all_predictions = 0, 0
        for current_labels, current_predictions in zip(list_of_labels, predictions):
            if not isinstance(current_labels, list):
                current_labels = [current_labels]

            all_labels += len(current_labels)
            all_predictions += len(current_predictions)
            for current_label in current_labels:
                if current_label in current_predictions:
                    recall += 1
            for current_prediction in current_predictions:
                if current_prediction in current_labels:
                    precision += 1

        return len(list_of_texts), round(100 * precision / all_predictions, 2), round(100 * recall / all_labels, 2)

    def test_file(self, test_data_path, k=1, threshold=-0.1, batch_size=1000, show_progress=True):
        """
        Predict top k predictions on given texts
        :param test_data_path: str, path to test file
        :param k: int, top k predictions
        :param threshold: float, from 0 to 1, default -0.1 meaining no threshold
        :param batch_size: int
        :param show_progress: bool
        :return: top k predictions and probabilities
        """
        data, labels = parse_txt(test_data_path, label_prefix=self.label_prefix)
        return self.test(data, labels, batch_size=batch_size, k=k, threshold=threshold, show_progress=show_progress)

    def export_model(self, destination_path):
        """
        Extract all the needed files for model loading to the specified destination.
        Also copies the training and validation files if available
        :param destination_path: str
        :return: None
        """
        all_paths = [value for key, value in self.info.items() if "path" in key]
        if "train_path" in self.hyperparams:
            all_paths.append(self.hyperparams["train_path"])

        if "test_path" in self.hyperparams:
            all_paths.append(self.hyperparams["test_path"])

        if "original_train_path" in self.hyperparams:
            all_paths.append(self.hyperparams["original_train_path"])
            all_paths.extend(self.hyperparams["additional_data_paths"])

        copy_all(all_paths, destination_path)
        model_params_path = os.path.join(destination_path, "model_params.json")
        with open(model_params_path, "r") as infile:
            model_params = json.load(infile)
        for key, value in model_params.items():
            if key.endswith("path"):
                model_params[key] = os.path.join(os.path.abspath(destination_path), value.split("/")[-1])
        with open(model_params_path, "w+") as outfile:
            json.dump(model_params, outfile)


class train_supervised(FastTextModel):
    def __init__(self, train_path, test_path=None, additional_data_paths=None, hyperparams=None,
                 preprocessing_function=None, log_dir="./", use_gpu=False, gpu_fraction=0.5, verbose=True,
                 remove_extra_labels=True, force=False):
        """
        Train a supervised fasttext model
        :param train_path: str, path to train file
        :param test_path: str or None, path to test file, if None training will be done without test
        :param additional_data_paths: list of str, paths of fasttext format additional data to concat with train file
        :param hyperparams: dict, all hyperparams for train_supervised
        :param preprocessing_function: function, function to apply on text data before feeding into network
        :param log_dir: str, directory to save the training files and the model
        :param use_gpu: bool, use gpu for training
        :param gpu_fraction: float, gpu fraction to allocate
        :param remove_extra_labels: bool, remove data from additional paths, which have labels not contained in
            train.txt
        :param verbose: bool
        :param remove_extra_labels: bool, remove datapoints with labels which appear in additional_data_paths but not in
        train_data_path. Ignored if additional_data_paths is None
        :param force: bool, forced training
        :return: object, the trained model
        """
        log_dir = validate(log_dir)

        # defualt hyperparams
        self.hyperparams = \
            {"train_path": '',
             "test_path": '',
             "label_prefix": "__label__",
             "data_fraction": 1,
             "seed": 17,
             "embedding_dim": 100,
             "num_epochs": 10,
             "word_ngrams": 1,
             "sort_ngrams": 0,
             "batch_size": 4096,
             "use_batch_norm": 0,
             "min_word_count": 1,
             "learning_rate": 0.1,
             "learning_rate_multiplier": 0.8,
             "dropout": 0.5,
             "l2_reg_weight": 1e-06,
             "batch_size_inference": 4096,
             "top_k": 3,
             "compare_top_k": 0,
             "save_all_models": 0,
             "use_test": 0,
             "use_gpu": 0,
             "gpu_fraction": 0.5,
             "cache_dir": handle_space_paths(os.path.abspath(os.path.join(log_dir, "cache"))),
             "log_dir": handle_space_paths(os.path.abspath(os.path.join(log_dir, "results"))),
             "force": 0,
             "progress_bar": 1,
             "flush": 1}

        if not os.path.exists(train_path):
            raise FileNotFoundError("train_path is incorrect")
        if test_path:
            if not os.path.exists(test_path):
                raise FileNotFoundError("test_path is incorrect")

        if preprocessing_function and verbose:
            print("Preprocessing train data ...")
        to_restore = dict()

        if hyperparams is None:
            hyperparams = dict()

        do_preprocessing = preprocessing_function is not None

        if len(hyperparams) != 0:
            for key, value in hyperparams.items():
                if key not in self.hyperparams:
                    to_restore[key] = value
                    print("WARNING! {} not in hyperparams, ignoring it".format(key))
                else:
                    if key in ["cache_dir", "log_dir"]:
                        self.hyperparams[key] = handle_space_paths(value)
                    else:
                        self.hyperparams[key] = value
        train_path = os.path.abspath(train_path)
        if additional_data_paths:
            data_to_save = []
            paths_joined_hashed = hash_(" ".join(additional_data_paths))
            concat_path = "./tmp.txt"
            joined_path = "./{}.txt".format(paths_joined_hashed)
            _, all_labels = parse_txt(train_path)
            unique_labels = np.unique(all_labels)
            if not isinstance(additional_data_paths, list):
                raise ValueError("Type of additional_data_paths should be list")
            for additional_data_path in additional_data_paths:
                if not os.path.isfile(additional_data_path):
                    raise FileNotFoundError("{} in additional data paths doesn't exist".format(additional_data_path))
                current_data, current_labels = parse_txt(additional_data_path)
                if remove_extra_labels:
                    needed_mask = np.in1d(current_labels, unique_labels)
                    current_data = [data for data, needed in zip(current_data, needed_mask) if needed]
                    current_labels = [data for data, needed in zip(current_labels, needed_mask) if needed]
                if do_preprocessing:
                    data_to_save.extend(["{}{} {}".format(self.hyperparams["label_prefix"], label,
                                                          preprocessing_function(data)) for label, data
                                         in zip(current_labels, current_data)])
                else:
                    data_to_save.extend(["{}{} {}".format(self.hyperparams["label_prefix"], label, data) for label, data
                                         in zip(current_labels, current_data)])
            np.savetxt(concat_path, data_to_save, fmt="%s")
            if do_preprocessing:
                prep_train_path = preprocess_data(train_path, preprocessing_function)
                os.system("cat {} {} > {}".format(concat_path, prep_train_path, joined_path))
                to_restore["original_train_path"] = prep_train_path
            else:
                os.system("cat {} {} > {}".format(concat_path, train_path, joined_path))
                to_restore["original_train_path"] = train_path
            self.hyperparams["train_path"] = joined_path
            to_restore["additional_data_paths"] = additional_data_paths
        else:
            if do_preprocessing:
                prep_train_path = preprocess_data(train_path, preprocessing_function)
                self.hyperparams["train_path"] = prep_train_path
            else:
                self.hyperparams["train_path"] = train_path

        if preprocessing_function and verbose:
            print("Done!")

        if test_path is not None:
            test_path = os.path.abspath(test_path)
            self.hyperparams["use_test"] = 1
            if do_preprocessing:
                prep_test_path = preprocess_data(test_path, preprocessing_function)
                to_restore["original_test_path"] = test_path
                self.hyperparams["test_path"] = prep_test_path
            else:
                self.hyperparams["test_path"] = test_path

        if use_gpu:
            self.hyperparams["use_gpu"] = 1
            self.hyperparams["gpu_fraction"] = gpu_fraction

        if force:
            self.hyperparams["force"] = 1

        # using Popen as calling the command from Jupyter doesn't deallocate GPU memory
        train_command = self._get_train_command()
        process = Popen(train_command, stdout=PIPE, shell=True, stderr=STDOUT, bufsize=1, close_fds=True)
        self.top_1_accuracy, self.top_k_accuracy, log_dir = \
            get_accuracy_log_dir(process, self.hyperparams["top_k"], verbose)

        for key, value in to_restore.items():
            self.hyperparams[key] = value
        super(train_supervised, self).__init__(model_path=os.path.join(log_dir, "model_best.pb"),
                                               model_params_path=os.path.join(log_dir, "model_params.json"),
                                               use_gpu=use_gpu, gpu_fraction=gpu_fraction, hyperparams=self.hyperparams,
                                               label_prefix=self.hyperparams["label_prefix"],
                                               preprocessing_function=preprocessing_function)

    def _get_train_command(self):
        args = ["--{} {}".format(key, value) for key, value in self.hyperparams.items() if str(value)]
        current_directory = os.path.dirname(inspect.getfile(inspect.currentframe()))
        train_command = " ".join(["python3 {}".format(os.path.join(current_directory, "main.py"))] + args)
        return train_command
