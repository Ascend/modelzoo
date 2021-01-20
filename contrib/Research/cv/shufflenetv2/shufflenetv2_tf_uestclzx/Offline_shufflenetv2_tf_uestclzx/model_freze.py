import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# from npu_bridge.estimator import npu_ops
from architecture import shufflenet_v2
# 导入网络模型文件
# 指定checkpoint路径
ckpt_path = "./model/model.ckpt-1601408"

def main():
    tf.reset_default_graph()
    # 定义网络的输入节点
    features = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # 调用网络模型生成推理图
    is_training = False
    logits = shufflenet_v2(
        features, is_training,
        num_classes=1000,
        depth_multiplier='0.5'
    )
    # 定义网络的输出节点
    logits = tf.nn.softmax(logits, axis=1, name="logits")
    print(logits)
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    print(predict_class)
    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './test_pb_model', 'model.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(input_graph='./pb_model/model.pb',   # 传入write_graph生成的模型文件
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
                                  output_node_names='output',  # 与定义的推理网络输出节点保持一致
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='./test_pb_model/shufflenetv2-1601408.pb',   # 改为需要生成的推理网络的名称
                                  clear_devices=False,
                                  initializer_nodes=''
                                  )
    print("done")

if __name__ == '__main__':
    main()
