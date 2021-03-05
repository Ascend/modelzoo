import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
import os

# 导入网络模型
import i3d_new_bn as i3d

def main():
    tf.reset_default_graph()
    
    # 定义网络的输入节点
    inputs = tf.placeholder(tf.float32, shape=[None, None, 224, 224, 2], name='input')
    
    # 调用网络模型生成推理图
    
    rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(inputs, is_training=False, dropout_keep_prob=1.0)
    rgb_logits_dropout = npu_ops.dropout(rgb_logits, 1.0)
# #     # 添加全联接层，使输出分类数量为101
    rgb_fc_out = tf.layers.dense(rgb_logits_dropout, 101, use_bias=True)
    
#     rgb_top_1_op = tf.nn.in_top_k(rgb_fc_out, label_holder, 1)
    # 定义预测结果
    predict_class = tf.argmax(rgb_fc_out, axis=1, output_type=tf.int32, name="final_output")
    
    # 保存训练图
    saver = tf.train.Saver()

    
    # 初始化参数
    sess = tf.Session()
    init = tf.global_variables_initializer() 
    sess.run(init)
    
    

    
    #with tf.Session() as sess:
        #保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
    tf.train.write_graph(sess.graph_def, '../pb_model_f', 'model.pb')    # 通过write_graph生成模型文件
    freeze_graph.freeze_graph(
		       input_graph='../pb_model_f/model.pb',   # 传入write_graph生成的模型文件
		       input_saver='',
		       input_binary=False, 
		       input_checkpoint='../data/checkpoints2/f/new/63300_0.962_model-63300',  # 传入训练生成的checkpoint文件
		       output_node_names='final_output',  # 与定义的推理网络输出节点保持一致
		       restore_op_name='save/restore_all',
		       filename_tensor_name='save/Const:0',
		       output_graph='../pb_model_f/model_flow.pb',   # 改为需要生成的推理网络的名称
		       clear_devices=False,
		       initializer_nodes='')
    print("Convert ckpt to pb successfully!")

def rename_var(ckpt_path, new_ckpt_path):
    """修改训练参数名称"""
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            #print('var_name---:', var_name)
            var = tf.contrib.framework.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('Flow/inception_i3d', 'inception_i3d')
            new_var_name = new_var_name.replace('batch_norm', 'BatchNorm')
            new_var_name = new_var_name.replace('Flow/dense','dense')
            var = tf.Variable(var, name=new_var_name)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)
        
def read_ckpt(ckpt_path):
    """阅读模型参数名称"""
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            print('ckpt_name: ', var_name)
            
if __name__ == '__main__': 
    # 定义保存后的模型路径
    ckpt_path = '../data/checkpoints2/f/63300_0.962_model-63300'
#     ckpt_path = '../data/checkpoints2/14300_0.900_model-14300'
    new_ckpt_path = '../data/checkpoints2/f/new/63300_0.962_model-63300'
#     # 修改模型参数名称
    rename_var(ckpt_path, new_ckpt_path)

    print('*' * 30)
    # 输出模型参数名称
    read_ckpt(ckpt_path)
    
    print('*' * 30)
    read_ckpt(new_ckpt_path)
    # 运行主函数
    main()