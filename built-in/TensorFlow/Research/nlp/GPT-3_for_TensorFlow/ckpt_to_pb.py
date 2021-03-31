import tensorflow as tf
from tensorflow.python.tools import freeze_graph
 
from npu_bridge.npu_init import *

# 导入网络模型文件
from gpt import megatron
# 指定checkpoint路径
ckpt_path = "/autotest/CI_daily/ModelZoo_GPT-3_TF/result/cloud-localhost-20210128143851-0/0/results/megatron_result/model.ckpt-0"

import megatron_config
params = megatron_config.megatron_config()
params['hidden_size'] = params['n_embd']

def main(): 
    tf.reset_default_graph()
    # 定义网络的输入节点
    inputs = tf.placeholder(tf.int32, shape=[None, 512], name="input")
    # 调用网络模型生成推理图
    logits = megatron(features=inputs, params=params, past=None, is_training=False)
    # 定义网络的输出节点
    #logits = tf.cast(logits, tf.float32)
    predict_class = tf.identity(logits, name="output")

    with tf.Session() as sess:
        #保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
		        input_graph='./pb_model/model.pb',   # 传入write_graph生成的模型文件
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
		        output_node_names='output',  # 与定义的推理网络输出节点保持一致
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='./gpt3_8p.pb',   # 改为需要生成的推理网络的名称
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()
