import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# from npu_bridge.estimator import npu_ops
from network import Deeplab_v2

ckpt_path = "./model/model.ckpt-2000"



def main():
    tf.reset_default_graph()
    
    X = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="input")
    
    is_training = tf.constant(False,dtype=tf.bool)
    
    net = Deeplab_v2(X, 21, False)
    raw_output = net.outputs
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(X)[1:3, ])
    raw_output = tf.argmax(raw_output, axis=3)
    pred = tf.expand_dims(raw_output, dim=3, name="output")
    
    with tf.Session() as sess:
        
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')   
        freeze_graph.freeze_graph(input_graph='./pb_model/model.pb',  
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_path,  
                                  output_node_names='output',  
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='./pb_model/deeplabv2.pb', 
                                  clear_devices=False,
                                  initializer_nodes=''
                                  )
    print("done")

if __name__ == '__main__':
    main()
