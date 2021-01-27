import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from architecture import shufflenet_v2

# checkpoint path
ckpt_path = "./model/model.ckpt"

def main():
    tf.reset_default_graph()
    # Input Node
    features = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # Inference graph
    is_training = False
    logits = shufflenet_v2(
        features, is_training,
        num_classes=1000,
        depth_multiplier='0.5'
    )
    # Output Node
    logits = tf.nn.softmax(logits, axis=1, name="logits")
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    with tf.Session() as sess:
        
        tf.train.write_graph(sess.graph_def, './test_pb_model', 'model.pb')    
        freeze_graph.freeze_graph(input_graph='./pb_model/model.pb',   
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_path,  
                                  output_node_names='output',  
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='./test_pb_model/shufflenetv2.pb',   
                                  clear_devices=False,
                                  initializer_nodes=''
                                  )
    print("done")

if __name__ == '__main__':
    main()
