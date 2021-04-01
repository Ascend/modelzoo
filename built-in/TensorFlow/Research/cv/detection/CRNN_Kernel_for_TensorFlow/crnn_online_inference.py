# -*- coding: utf-8 -*-
#
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

import tensorflow as tf
import os
import glog as log
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
import time
import numpy as np
import tqdm

from tools import other_dataset_evaluate_shadownet as evaluateShadownet
from local_utils import evaluation_tools
from tools import evaluate_shadownet 
from data_provider import tf_io_pipline_fast_tools

# 用户自定义模型路径、输入、输出参数的
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='./shadownet_tf.pb',help="""pb path""")
    parser.add_argument('--image_path', default = './data/test/IIIT5K/',help = """the bert dev.json dir path""")
    #parser.add_argument('--output_dir', default = '../data/',help = """output_dir""")
    #parser.add_argument('--pre_process', default = "True",help = """do pre process""")
    #parser.add_argument('--post_process', default = "True",help = """do pre process""")
    parser.add_argument('--batchsize', default = 64,help = """在线推理的batchsize""")
    parser.add_argument('--input_tensor_name', default = 'input:0',help = """input_tensor_name""")
    parser.add_argument('--output_tensor_name', default = 'shadow_net/sequence_rnn_module/transpose_time_major:0',help = """input_tensor_name""")
    parser.add_argument('--annotation_file', default='./data/test/IIIT5K/annotation.txt',help="""label file""")
    parser.add_argument('--char_dict_path', default='./data/char_dict/char_dict.json',help="""char_dict_path""")
    parser.add_argument('--ord_map_dict_path',default='./data/char_dict/ord_map.json',help="""ord_map_dict_path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


#在线推理配置   
class Classifier(object):
    args = parse_args()
    def __init__(self):
        # 昇腾AI处理器模型编译和优化配置
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        
        # 配置1： 选择在昇腾AI处理器上执行推理
        custom_op.parameter_map["use_off_line"].b = True

        # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
        custom_op.parameter_map["graph_run_mode"].i = 0

        # 配置4：开启profiling，需要时配置，不需要时注释掉
        #custom_op.parameter_map["profiling_mode"].b = True
        #custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes("task_trace")

        # 配置4：关闭remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # --------------------------------------------------------------------------------

        # 加载模型，并指定该模型的输入和输出节点
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        加载用于推理fronzen graph模型
        如果加载其他的类型的模型文件（saved model/check point），可按照对应类型的加载方法
        :param model_file:
        :return:
        """
        print('****Enter load_model:', model_file)
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
        print('**** ParseFromString')
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        print('**** import_graph_def')
        return graph
    
    def do_infer(self, batch_data):
        """
        执行推理, 推理输入的shape必须保持一致
        :param image_data:
        :return:
        """
        total_time = 0
        t = time.time()
        out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_data})
        total_time = time.time() - t
        print('[INFO] inference time is {}', total_time)
        return np.array(out), total_time


def main():
    args = parse_args()
    top1_count = 0
    top5_count = 0

    log.info('Start predicting...')
    per_char_accuracy = 0
    full_sequence_accuracy = 0.0
    total_labels_char_list = []
    total_predictions_char_list = []

    tf.reset_default_graph()
    print("########NOW Start loadmodel!!!#########")
    print("args.model_path {}", args.model_path)
    classifier = Classifier()

    ###data preprocess
    print("########NOW Start Preprocess!!!#########")
    annotation_list = evaluateShadownet.get_annotation(args.annotation_file)
    num_iterations = len(annotation_list) // args.batchsize
    epoch_tqdm = tqdm.tqdm(range(num_iterations))

    out_list = []
    total_time = 0
    for i in epoch_tqdm:
        # for i in range(num_iterations):
        anns = annotation_list[i * args.batchsize:(i + 1) * args.batchsize]
        batch_data, batch_label = evaluateShadownet.get_batch_data(args.image_path, anns)

        test_predictions_value, total_time = classifier.do_infer(batch_data)
        decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path)

        #test_predictions_value = evaluateShadownet.decoder.sparse_tensor_to_str(test_predictions_value[0])
        test_predictions_value = decoder.sparse_tensor_to_str(test_predictions_value[0])
        #test_predictions_value = evaluate_shadownet.decoder.sparse_tensor_to_str(test_predictions_value[0])

        #test_predictions_value.append(test_predictions_value)
        total_time += total_time

        per_char_accuracy += evaluation_tools.evaluation_tools.compute_accuracy(
            batch_label, test_predictions_value, display=False, mode='per_char'
        )

        full_sequence_accuracy += evaluation_tools.evaluation_tools.compute_accuracy(
            batch_label, test_predictions_value, display=False, mode='full_sequence'
        )
        for index, ann in enumerate(anns):
            log.info(ann)
            log.info("predicted values :{}".format(test_predictions_value[index]))

    epoch_tqdm.close()
    avg_per_char_accuracy = per_char_accuracy / num_iterations
    avg_full_sequence_accuracy = full_sequence_accuracy / num_iterations
    log.info('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
    log.info('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))
    print('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
    print('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))


if __name__ == '__main__':
    main()




