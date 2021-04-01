# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-01 22:28:41
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-02 18:36:50
# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility functions for GLUE classification tasks."""

# 通过加载已经训练好的pb模型，执行推理
import numpy as np
import time
import sys,os
import shutil
from scipy.io import wavfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_bridge.estimator import npu_ops
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import argparse
from params import hparams

# 用户自定义模型路径、输入、输出
input_tensor_name = 'lc_infer:0'
output_tensor_name = 'Waveglow/Reshape_2:0'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='./frozen_model.pb',
                        help="""pb path""")
    parser.add_argument('--data_dir', default = './datasets/binfile_5',
                        help = """the bin dir path""")
    parser.add_argument('--output_dir', default = './datasets/',
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
    return v.lower() in ("yes","true","t","1")

def write_wav(waveform, sample_rate, filename):
    """

    :param waveform: [-1,1]
    :param sample_rate:
    :param filename:
    :return:
    """
    # TODO: write wave to 16bit PCM, don't use librosa to write wave
    y = np.array(waveform, dtype=np.float32)
    y *= 32767
    wavfile.write(filename, sample_rate, y.astype(np.int16))
    print('Updated wav file at {}'.format(filename))


class onlineInference(object):

    def __init__(self,model_path,batchSize):
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
        # custom_op.parameter_map["profiling_mode"].b = True
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/tmp/profiling","task_trace":"on"}')

        # 配置5：关闭remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # --------------------------------------------------------------------------------
        print('****before load model')
        # 加载模型，并指定该模型的输入和输出节点
        self.graph = self.__load_model(model_path)
        self.batch_size = batchSize
        print('****After load model')

        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
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

    def do_infer(self, model_path, data_dir):
        '''
        :param pb_path:pb文件的路径
        :param bin_path:bin文件的路径
        :return:
        '''
        out_list = []
        cost_time = 0
        filelist = []
        for file in os.listdir(data_dir):
            filepath = os.path.join(data_dir, file)
            print(filepath)
            filelist.append(file)

            # bin文件转mel频谱图
            input_temp = np.fromfile(filepath, dtype=np.float32)
            mel_spec = input_temp.reshape((1, -1, 80))
            print(mel_spec)

            start_time = time.time()
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: mel_spec})
            
            audio_output = out[0].flatten()

            print(file)
            if file != filelist[0]:
                end_time = time.time()
                cost_time += end_time - start_time
                print(cost_time * 1000 / (len(filelist) - 1))
            write_wav(audio_output, hparams.sample_rate,
                      "./result_wav/waveglow_online_" + str(file).split(".")[0] + ".wav")
            print("success construct one audio...")
        self.sess.close()
        return cost_time, filelist

if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    online = onlineInference(args.model_path, args.batchSize)

    cost_time, filelist = online.do_infer(args.model_path, args.data_dir)

    print("cost time:", cost_time)
    #print(len(filelist))
    print("average infer time:{0:0.3f} ms/wav,FPS is {1:0.3f}".format(cost_time * 1000 / (len(filelist)-1), 1 / (cost_time / (len(filelist)-1))))
    fo = open("./perform_static_dev_0_chn_0.txt", "w")
    fo.write("average infer time: {0:0.3f} ms {1:0.3f} fps/s".format(cost_time * 1000 / (len(filelist)-1), 1 / (cost_time / (len(filelist)-1))))
    fo.close()
