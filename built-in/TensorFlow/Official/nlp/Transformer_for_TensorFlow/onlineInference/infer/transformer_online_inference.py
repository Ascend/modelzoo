# transformer end to end online inference

# -*- coding: utf-8 -*-
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for GLUE classification tasks."""

# 通过加载已经训练好的pb模型，执行推理
import time
import sys
import os
import argparse
import shutil
import numpy as np
from npu_bridge.estimator import npu_ops
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/pre_treatment')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/post_treatment')
from pre_treatment.transformer_create_infer_data import *
from pre_treatment.transformer_create_infer_data import create_infer_input_data
from post_treatment.transformer_calculation_bleu_score import *
from post_treatment.transformer_calculation_bleu_score import post_process
#from tensorflow.python import debug as tf_debug

# 用户自定义模型路径、输入、输出
input_tensor_name1 = 'source_ids:0'
input_tensor_name2 = 'source_mask:0'
output_tensor_name = 'predicted_ids:0'


def parse_args():


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='../model/transformer_infer.pb',
                        help="""pb path""")
    parser.add_argument('--data_dir', default='../datasets',
                        help="""the bert dev.json dir path""")
    parser.add_argument('--output_dir', default='../datasets/',
                        help="""output_dir""")
    parser.add_argument('--pre_process', default="False",
                        help="""do pre process""")
    parser.add_argument('--post_process', default="True",
                        help="""do pre process""")
    parser.add_argument('--batchSize', default=1,
                        help="""在线推理的batchSize""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def str2bool(v):

    return v.lower() in ("yes", "true", "t", "1")


class OnlineInference(object):    


    def __init__(self, model_path, batch_size):
        # 昇腾AI处理器模型编译和优化配置
        # --------------------------------------------------------------------------------
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 配置1： 选择在昇腾AI处理器上执行推理
        custom_op.parameter_map["use_off_line"].b = True

        # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
        # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
        #custom_op.parameter_map["mix_compile_mode"].b = True  # 开启混合计算
        #custom_op.parameter_map["dynamic_input"].b = True
        #custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
        # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
        custom_op.parameter_map["graph_run_mode"].i = 0

        # 配置4：开启profiling，需要时配置，不需要时注释掉
        #custom_op.parameter_map["profiling_mode"].b = True
        #custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes("task_trace")

        #配置5：Dump数据：
        ## 是否开启dump功能
        #custom_op.parameter_map["enable_dump"].b = True
        ## dump数据存放路径
        #custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/data/zhangsz/transformer_online_dump/")
        ## dump模式，默认仅dump算子输出数据，还可以dump算子输入数据，取值：input/output/all
        #custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

        # 配置6：关闭remapping
        config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
        # --------------------------------------------------------------------------------
        print('****before load model')
        # 加载模型，并指定该模型的输入和输出节点
        cost_time = 0
        start_time = time.time()
        self.graph = self.__load_model(model_path)
        end_time = time.time()
        self.batch_size = batch_size
        print('****After load model')
        self.input_tensor1 = self.graph.get_tensor_by_name(input_tensor_name1)
        self.input_tensor2 = self.graph.get_tensor_by_name(input_tensor_name2)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)
        cost_time = end_time-start_time
        print("Load model cost time:", cost_time)

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


    def do_infer(self, input_ids, input_reshapes, file_name, infer_out_dir):
        """
        online infer,the shape of input must be consistent
        :param image_data:
        :return:
        """
        out_list = []
        cost_time = 0
        print("****len(input_ids):", len(input_ids))
		    #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type="readline")
        for i in range(len(input_ids)):
            start_time = time.time()
            print("****infer i:", i)
            print("input_tensor1:", input_ids[i])
            print("input_tensor2:", input_reshapes[i])
            #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type="readline")
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor1: input_ids[i],self.input_tensor2: input_reshapes[i]})
            if i != 0 :
                end_time = time.time()
                cost_time += end_time-start_time
            for j in range(out.shape[0]):
                if i*int(self.batch_size)+j < len(file_name) :
                    print("===infer input name:", file_name[i*int(self.batch_size)+j])
                    name = file_name[i*int(self.batch_size)+j][0:file_name[i*int(self.batch_size)+j].rfind(".")]
                    out[j].tofile(infer_out_dir + "/davinci_" + name + "_output0.bin")
                    print("****infer i:", i, "save result")
        self.sess.close()
        return cost_time


    def batch_process(self, source_path, input_index):
        """
        图像预处理，加载数据集
        :return:
        """
        inputs_src = []
        inputs_tgt = []
        filelist=[]
        input_len = 128
        print("*****input_len:", input_len)
        batch_size = int(self.batch_size)
        file_seg_str = "_".join(os.listdir(source_path)[0].split("_")[:-1])
        print("******file_seg_str:", file_seg_str)
        for i in range(len(os.listdir(source_path))):
            file = file_seg_str + "_" + str(i) + ".bin"
            filepath = os.path.join(source_path, file)
            input_temp1 = np.fromfile(filepath, dtype=np.int32)
            input_temp1 = input_temp1.reshape((128))

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
                pad = np.zeros((128)).astype(np.int32)
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
    infer_source_path = output_dir
    real_file = data_dir + "/newstest2014.tok.bpe.32000.de"
    infer_out_dir = "./result_Files"
    vocab_file = data_dir + "/vocab.share"
    pre_source_file = data_dir + "/newstest2014.tok.bpe.32000.en"
    mind_file = data_dir + "/newstest2014-l128-mindrecord"
    # pre process
    if str2bool(args.pre_process):
        if (os.path.exists(infer_source_path)):
            shutil.rmtree(infer_source_path)
        os.mkdir(infer_source_path)
        pre_all_data_path = data_dir + "/test.all"
        if os.path.exists(pre_all_data_path):
            os.remove(pre_all_data_path)
        os.system('paste %s %s  > %s' % (pre_source_file, real_file, pre_all_data_path))
        create_infer_input_data(pre_all_data_path, vocab_file, mind_file, 128, False, 16, infer_source_path)
    # online infer
    if os.path.exists(infer_out_dir):
        shutil.rmtree(infer_out_dir)
    os.mkdir(infer_out_dir)
    online = OnlineInference(args.model_path, args.batchSize)

    input_ids, filelist = online.batch_process(infer_source_path + "/source_ids", 1)
    input_reshapes, filelist = online.batch_process(infer_source_path + "/source_masks", 2)
    cost_time = online.do_infer(input_ids, input_reshapes, filelist, infer_out_dir)

    # post process
    if str2bool(args.post_process):
        file_bleu = "./infer_out_for_bleu"
        if os.path.exists(file_bleu):
            os.remove(file_bleu)
        post_process(infer_out_dir, real_file, vocab_file, file_bleu)
    print("cost time:", cost_time)
    print("average infer time:{0:0.3f} ms/img,FPS is {1:0.3f}".format(cost_time * 1000 / (len(filelist) - 1*int(args.batchSize)),
                                                                      1 / (cost_time / (len(filelist)-1*int(args.batchSize)))))
