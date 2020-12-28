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

"""Utility functions for GLUE classification tasks."""

# 通过加载已经训练好的pb模型，执行推理
import numpy as np
import time
import sys,os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_bridge.estimator import npu_ops
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import argparse
#from pre_treatment.jasper_prep_process import *
from post_treatment.jasper_post_process import *

# 用户自定义模型路径、输入、输出
input_tensor_name1 = 'input:0'
input_tensor_name2 = 'input_reshape:0'
output_tensor_name = 'ForwardPass/fully_connected_ctc_decoder/logits:0'
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
    return v.lower() in ("yes","true","t","1")

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
        #custom_op.parameter_map["profiling_mode"].b = True
        #custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes("task_trace")

        # 配置5：关闭remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # --------------------------------------------------------------------------------
        print('****before load model')
        # 加载模型，并指定该模型的输入和输出节点
        cost_time = 0
        start_time = time.time()
        self.graph = self.__load_model(model_path)
        end_time = time.time()
        self.batch_size = batchSize
        print('****After load model')
        self.input_tensor1 = self.graph.get_tensor_by_name(input_tensor_name1)
        self.input_tensor2 = self.graph.get_tensor_by_name(input_tensor_name2)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)
        cost_time = end_time-start_time
        saveFimeName = "./load_model_perfor.txt"
        with open(saveFimeName, 'w', encoding='utf-8') as f:
            saveText = "Load model cost time:" + str(cost_time)
            f.write(saveText)
        #self.sess.close()
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

    def do_infer(self, input_ids,input_reshapes,fileName):
        """
        执行推理, 推理输入的shape必须保持一致
        :param image_data:
        :return:
        """
        out_list = []
        cost_time=0
        print("****len(input_ids):", len(input_ids))
        for i in range(len(input_ids)):
            start_time = time.time()
            print("****infer i:", i)
            print("input_tensor1:", input_ids[i])
            print("input_tensor2:", input_reshapes[i])
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor1: input_ids[i],self.input_tensor2: input_reshapes[i]})
            end_time = time.time()
            cost_time +=end_time-start_time
            for j in range(out.shape[0]):
                if i*int(self.batch_size)+j < len(fileName) :
                    name = fileName[i*int(self.batch_size)+j][0:fileName[i*int(self.batch_size)+j].rfind(".")]
                    out[j].tofile("./result_Files/davinci_"+name+"_output0.bin")
                    print("****infer i:", i, "save result")
        self.sess.close()
        return cost_time


    def batch_process(self, sourcePath, inputIndex):
        """
        图像预处理，加载数据集
        :return:
        """
        inputs_src = []
        inputs_tgt = []
        filelist=[]
        input_len = 0
        if inputIndex == 1 :
            input_len = 2336
        else :
            input_len = 1168
        print("*****input_len:", input_len)
        batch_size = int(self.batch_size)
        for file in os.listdir(sourcePath):
            filepath = os.path.join(sourcePath, file)
            input_temp1 = np.fromfile(filepath, dtype=np.float16)
            if inputIndex == 1 :
                input_temp1 = input_temp1.reshape((1, 2336, 64))
            else :
                input_temp1 = input_temp1.reshape((1168))
            #print("input_temp1.shape:", input_temp1.shape)
            inputs_src.append(input_temp1)
            filelist.append(file)
            if 0 == (len(inputs_src) % batch_size) :
                inputs_tgt.append(inputs_src)
                inputs_src = []

        m = len(inputs_src) % batch_size
        print("****input_src len:", len(inputs_src), "batch_size:", batch_size, "m:", m)
        if(m < batch_size and m > 0):
            print("****need pad data")
            for i in range(int(batch_size-m)) :
                if inputIndex == 1:
                    pad = np.zeros((1, 2336, 64)).astype(np.float16)
                else :
                    pad = np.zeros((1168)).astype(np.float16)
                inputs_src.append(pad)
            inputs_tgt.append(inputs_src)
            #print("*****pad:", pad,)
            #print("*****inputs_src:", inputs_src)
            #inputs_src = np.concatenate((inputs_src, pad), axis=1)
        '''
        i=0
        while i<len(inputs_src)-batch_size:
            inputs_tgt.append(inputs_src[i: i + batch_size])
            i += batch_size
        '''
        print("****inputs_tgt len:", len(inputs_tgt))
        return inputs_tgt,filelist


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    source_path = output_dir + "/jasper"
    real_file = data_dir + "/librivox-dev-clean.csv"
    #pre process
    if str2bool(args.pre_process):
        if(os.path.exists(source_path)):
            shutil.rmtree(source_path)
        pre_process(data_dir, source_path)
    #online infer

    if os.path.exists("./result_Files/"):
        shutil.rmtree("./result_Files/")
    os.mkdir("./result_Files")

    online = onlineInference(args.model_path,args.batchSize)
    input_ids,filelist=online.batch_process(source_path+"/input_0", 1)
    input_reshapes,filelist=online.batch_process(source_path+"/input_reshape", 2)

    cost_time=online.do_infer(input_ids,input_reshapes,filelist)

    #post process
    if str2bool(args.post_process):
        calc_jasper_infer_accuracy("./result_Files/", real_file)
    print("cost time:", cost_time)
    print("average infer time:{0:0.3f} ms/img,FPS is {1:0.3f}".format(cost_time*1000/len(filelist),1/(cost_time/len(filelist))))
