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
# =============================================================================
import argparse
import os.path as ops
import os
import math
import time
import sys
import tensorflow as tf
import numpy as np
import glog as log

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)

from config import global_config
from data_provider import tf_io_pipline_fast_tools
from local_utils import evaluation_tools


CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_dir', type=str,default='./output_fp32',
                        help='Directory of npu predict output')
    parser.add_argument('-c', '--char_dict_path', type=str,default='data/char_dict/char_dict.json',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,default='data/char_dict/ord_map.json',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-l', '--label_dir', type=str, default='./labels',
                        help='Path of ground truth labels')
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help='batchsize of input dataset')

    return parser.parse_args()

def evaluate_shadownet(input_dir, label_dir, char_dict_path,
                       ord_map_dict_path,batchsize=64):
    # setup decoder
    decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )
    test_inference_ret = tf.placeholder(tf.float32, shape=[25,batchsize,37], name="test_inference_ret")
    test_decoded, test_log_prob = tf.nn.ctc_greedy_decoder(
        test_inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(batchsize),
        merge_repeated=True
    )

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    inputPath = input_dir
    labelPath = label_dir
    with sess.as_default():

        log.info('Start predicting...')

        per_char_accuracy = 0
        full_sequence_accuracy = 0.0
        filelist = os.listdir(inputPath)
        filelist.sort()
        i = 0

        for file in filelist:
        if file.endswith(".bin"):
            with open(os.path.join(labelPath, 'batch_label_'+str(i).rjust(3,'0')+'.txt'),'r') as f:
                data = f.read()
            batch_label = data.replace('\r','').replace('\n','').split(',')
            input_data = np.fromfile(os.path.join(inputPath,file),dtype="float32").reshape(25,batchsize,37)
            test_predictions_value = sess.run(test_decoded,feed_dict={test_inference_ret: input_data})
            test_predictions_value = decoder.sparse_tensor_to_str(test_predictions_value[0])

            per_char_accuracy += evaluation_tools.compute_accuracy(
                        batch_label, test_predictions_value, display=False, mode='per_char'
                    )

            full_sequence_accuracy += evaluation_tools.compute_accuracy(
                        batch_label, test_predictions_value, display=False, mode='full_sequence'
                    )
            i += 1

        avg_per_char_accuracy = per_char_accuracy / i
        avg_full_sequence_accuracy = full_sequence_accuracy / i
        log.info('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
        log.info('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))


if __name__ == '__main__':
    args = init_args()

    evaluate_shadownet(
        input_dir=args.input_dir,
        label_dir=args.label_dir,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        batchsize=args.batchsize
    )
