#!/usr/bin/env python
# coding=utf-8

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
# ============================================================================
"""LeNet online inference"""

# 通过加载已经训练好的pb模型，执行推理
import time
import sys
import os
import shutil
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from LeNet_preprocess import *


# 用户自定义模型路径、输入、输出
input_tensor_name1 = 'input_image:0'
input_tensor_name2 = 'input_label:0'

output_tensor_name = 'Mean_1:0'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='../model/jasper_infer_float32.pb',
                        help="""pb path""")
    parser.add_argument('--data_dir', default = '../datasets',
                        help = """the bert dev.json dir path""")
    parser.add_argument('--output_dir', default = '../datasets/',
                        help = """output_dir""")
    parser.add_argument('--pre_process', default = "False",
                        help = """do pre process""")
    parser.add_argument('--post_process', default = "True",
                        help = """do pre process""")
    parser.add_argument('--batchSize', default = 1,
                        help = """在线推理的batchSize""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class OnlineInference(object):    

    def __init__(self, model_path, batchSize):
        # 昇腾AI处理器模型编译和优化配置
        # --------------------------------------------------------------------------------
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

        # 配置5：关闭remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # --------------------------------------------------------------------------------
        print('****before load model')
        # 加载模型，并指定该模型的输入和输出节点

        self.graph = self.__load_model(model_path)
    
        self.batch_size = batchSize
        print('****After load model')
        self.input_tensor1 = self.graph.get_tensor_by_name(input_tensor_name1)
        self.input_tensor2 = self.graph.get_tensor_by_name(input_tensor_name2)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)
     
    def __load_model(self, model_file):
        print('****Enter load_model:', model_file)
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
        print('**** ParseFromString')
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        print('**** import_graph_def')
        return graph

    def do_infer(self, input_ids, input_reshapes, fileName): 
        """
        执行推理, 推理输入的shape必须保持一致
        :param image_data:
        :return:
        """

        acc = 0
        total = 0
        cost_time = 0
        print("****len(input_ids):", len(input_ids))
        for i in range(len(input_ids)):
            start_time = time.time()
            print("****infer i:", i)

            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor1: input_ids[i], self.input_tensor2: input_reshapes[i]})
            end_time = time.time()
            cost_time += end_time - start_time
            acc += out
            total += 1
            # for j in range(out.shape[0]):
            #     if i*int(self.batch_size)+j < len(fileName) :
            #         name = fileName[i*int(self.batch_size)+j][0:fileName[i*int(self.batch_size)+j].rfind(".")]
            #         out[j].tofile("./result_Files/davinci_"+name+"_output0.bin")
            #         print("****infer i:", i, "save result")
        self.sess.close()
        print('======acc : {}----total : {}'.format(acc, total))
        print('Final Online Inference Accuracy accuracy : ', round(acc / total, 4))
        return cost_time

    def batch_process(self, sourcePath, inputIndex):
        """
        图像预处理，加载数据集
        :return:
        """
        inputs_src = []
        inputs_tgt = []
        filelist=[]

        batch_size = int(self.batch_size)
        for file in os.listdir(sourcePath):
            filepath = os.path.join(sourcePath, file)
            if inputIndex == 1: 
                input_temp1 = np.fromfile(filepath, dtype=np.float32)
                input_temp1 = input_temp1.reshape([784])
            else: 
                input_temp1 = np.fromfile(filepath, dtype=np.float64)
                input_temp1 = input_temp1.reshape([10])
            #print("input_temp1.shape:", input_temp1.shape)
            inputs_src.append(input_temp1)
            filelist.append(file)
            if 0 == (len(inputs_src) % batch_size): 
                inputs_tgt.append(inputs_src)
                inputs_src = []

        remainder_cnt = len(inputs_src) % batch_size
        print("****input_src len:", len(inputs_src), "batch_size:", batch_size, "remainder_cnt:", remainder_cnt)
        if(remainder_cnt < batch_size and remainder_cnt > 0):
            print("****need pad data")
            for i in range(int(batch_size - remainder_cnt)): 
                if inputIndex == 1:
                    pad = np.zeros((784)).astype(np.float32)
                else: 
                    pad = np.zeros((10)).astype(np.float64)
                inputs_src.append(pad)
            inputs_tgt.append(inputs_src)
            #print("*****pad:", pad,)
            #print("*****inputs_src:", inputs_src)
            #inputs_src = np.concatenate((inputs_src, pad), axis=1)

        print("****inputs_tgt len:", len(inputs_tgt))
        return inputs_tgt, filelist


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    #pre process
    if str2bool(args.pre_process):
        if(os.path.exists(output_dir)):
            shutil.rmtree(output_dir)
        input_data_get(data_dir, output_dir)
    #online infer

    if os.path.exists("./result_Files/"):
        shutil.rmtree("./result_Files/")
    os.mkdir("./result_Files")

    online = OnlineInference(args.model_path, args.batchSize)
    input_images, filelist=online.batch_process(output_dir + "/images", 1)
    input_labels, filelist=online.batch_process(output_dir + "/labels", 2)

    cost_time=online.do_infer(input_images, input_labels, filelist)

    print("cost time:", cost_time)
    print("average infer time:{0:0.3f} ms/img,FPS is {1:0.3f}".format(cost_time * 1000 / len(filelist), 1 / (cost_time / len(filelist))))
