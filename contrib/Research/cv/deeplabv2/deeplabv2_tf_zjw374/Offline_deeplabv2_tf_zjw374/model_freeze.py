import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# from npu_bridge.estimator import npu_ops
from network import Deeplab_v2
# 导入网络模型文件
# 指定checkpoint路径
#ckpt_path = "./model/deeplab_resnet.ckpt"
ckpt_path = "./model/model.ckpt-2000"



def main():
    tf.reset_default_graph()
    # 定义网络的输入节点
    X = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="input")
    # 调用网络模型生成推理图
    is_training = tf.constant(False,dtype=tf.bool)
    # logits = _mapping(
    #         X, is_training, NUM_CLASSES,
    #         groups, dropout, complexity_scale_factor
    #     )
    # 定义网络的输出节点
    net = Deeplab_v2(X, 21, False)
    raw_output = net.outputs
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(X)[1:3, ])
    raw_output = tf.argmax(raw_output, axis=3)
    pred = tf.expand_dims(raw_output, dim=3, name="output")
    #spred = tf.reshape(pred, [-1, ])
    #logits = tf.nn.softmax(logits, axis=1)
    #predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(input_graph='./pb_model/model.pb',   # 传入write_graph生成的模型文件
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
                                  output_node_names='output',  # 与定义的推理网络输出节点保持一致
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='./pb_model/deeplabv2.pb',   # 改为需要生成的推理网络的名称
                                  clear_devices=False,
                                  initializer_nodes=''
                                  )
    print("done")

if __name__ == '__main__':
    main()
