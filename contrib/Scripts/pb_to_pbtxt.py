import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)

def convert_pbtxt_to_pb(filename):
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', 'protobuf.pb', as_text=False)

filepath = 'H:/SimpleTest/protobuf.pbtxt'
convert_pbtxt_to_pb(filepath)
#convert_pb_to_pbtxt(filepath)
