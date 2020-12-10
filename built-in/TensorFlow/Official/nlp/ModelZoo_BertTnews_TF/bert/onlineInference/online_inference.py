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
from pre_treatment.get_bert_tnews_infer_input_new import *
from post_treatment.calc_bert_accuracy import *

# 用户自定义模型路径、输入、输出
input_tensor_name1 = 'input_ids:0'
input_tensor_name2 = 'input_mask:0'
input_tensor_name3 = 'segment_ids:0'
output_tensor_name = 'loss/BiasAdd:0'
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='../model/bert_tnews/bert_tnews_graph.pb',
                        help="""pb path""")
    parser.add_argument('--data_dir', default = '../../../CLUEdataset/tnews/',
                        help = """the bert dev.json dir path""")
    parser.add_argument('--vocab_file', default='../../../CLUEdataset/tnews/vocab.txt',
                        help="""the bert vocab.txt dir path""")
    parser.add_argument('--output_dir', default = '../../../CLUEdataset/',
                        help = """output_dir""")
    parser.add_argument('--pre_process', default = "True",
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

        # 配置4：关闭remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # --------------------------------------------------------------------------------

        # 加载模型，并指定该模型的输入和输出节点
        self.graph = self.__load_model(model_path)
        self.batch_size = batchSize
        self.input_tensor1 = self.graph.get_tensor_by_name(input_tensor_name1)
        self.input_tensor2 = self.graph.get_tensor_by_name(input_tensor_name2)
        self.input_tensor3 = self.graph.get_tensor_by_name(input_tensor_name3)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        加载用于推理fronzen graph模型
        如果加载其他的类型的模型文件（saved model/check point），可按照对应类型的加载方法
        :param model_file:
        :return:
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, input_ids,input_masks,segments,fileName):
        """
        执行推理, 推理输入的shape必须保持一致
        :param image_data:
        :return:
        """
        out_list = []
        cost_time=0
        for i in range(len(input_ids)):
            start_time = time.time()          
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor1: input_ids[i],self.input_tensor2: input_masks[i],self.input_tensor3: segments[i]})
            end_time = time.time()
            cost_time +=end_time-start_time
            for j in range(out.shape[0]):
                name = fileName[i*int(self.batch_size)+j][0:fileName[i*int(self.batch_size)+j].rfind(".")]
                out[j].tofile("./result_Files/davinci_"+name+"_output0.bin")
        self.sess.close()
        return cost_time


    def batch_process(self, sourcePath):
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
            input_temp1 = np.fromfile(filepath, dtype=np.int32)
            inputs_src.append(input_temp1)
            filelist.append(file)
        m = len(inputs_src) % batch_size
        if(m < batch_size):
            pad = np.zeros((batch_size - m, 128)).astype(np.int32)
            inputs_src = np.concatenate((inputs_src, pad), axis=0)
        i=0
        while i<len(inputs_src)-batch_size:
            inputs_tgt.append(inputs_src[i: i + batch_size])
            i += batch_size
        return inputs_tgt,filelist


if __name__ == '__main__':
    task_name = "tnews"
    args = parse_args()
    data_dir = args.data_dir
    vocab_file = args.vocab_file
    output_dir = args.output_dir
    source_path = output_dir + "bert_tnews"
    #pre process
    if str2bool(args.pre_process):
        if(os.path.exists(source_path)):
            shutil.rmtree(source_path)
        preprocess(task_name, data_dir, vocab_file, output_dir)
    #online infer

    if os.path.exists("./result_Files/"):
        shutil.rmtree("./result_Files/")
    os.mkdir("./result_Files")

    online = onlineInference(args.model_path,args.batchSize)
    input_ids,filelist=online.batch_process(source_path+"/input_ids")
    input_masks,filelist=online.batch_process(source_path+"/input_masks")
    segments,filelist=online.batch_process(source_path+"/segments")
    
    cost_time=online.do_infer(input_ids,input_masks,segments,filelist)

    #post process
    if str2bool(args.post_process):
        calc_bert_infer_accuracy("./result_Files/", args.data_dir+"/dev.json",  args.data_dir+"/labels.json")
    print("cost time:", cost_time)
    print("average infer time:{0:0.3f} ms/img,FPS is {1:0.3f}".format(cost_time*1000/len(filelist),1/(cost_time/len(filelist))))
