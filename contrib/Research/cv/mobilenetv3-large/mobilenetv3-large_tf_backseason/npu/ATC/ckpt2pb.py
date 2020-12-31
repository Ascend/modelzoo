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

# borrow from https://support.huaweicloud.com/mprtg-A800_9000_9010/atlasprtg_13_0042.html
# 
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# from npu_bridge.estimator import npu_ops 
# from nets.mobilenet import mobilenet_v3
from nets import nets_factory

# 导入网络模型文件
# import alexnet
# 指定checkpoint路径

ckpt_path = "../gpu/results_mbv3_74_10/model.ckpt-1200000"
dst_folder = './ATC/pb_model'

# ckpt_path = "snapshots_official/model-540000"
# dst_folder = './ATC/pb_model_official'

def main(): 
    tf.compat.v1.reset_default_graph()
    # 定义网络的输入节点
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # 调用网络模型生成推理图
    # logits = mobilenet_v3.large.inference(inputs, version="he_uniform",
    #                               num_classes=1000, is_training=False)
    network_fn = nets_factory.get_network_fn(
        'mobilenet_v3_large',
        num_classes=1001,
        is_training=False)
    logits, end_points = network_fn(inputs)
    # 定义网络的输出节点
    # predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    predict_class = tf.identity(logits, name='output')
    with tf.compat.v1.Session() as sess:
        #保存图，在 dst_folder 文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.io.write_graph(sess.graph_def, dst_folder, 'tmp_model.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
		        input_graph=os.path.join(dst_folder, 'tmp_model.pb'),   # 传入write_graph生成的模型文件
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
		        output_node_names='output',  # 与定义的推理网络输出节点保持一致
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph=os.path.join(dst_folder, 'mobilenet_v3_large.pb'),   # 改为需要生成的推理网络的名称
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()
