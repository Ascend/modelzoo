from tensorflow.python.platform import gfile
import tensorflow as tf 
import numpy as np
import os 

sess = tf.Session()
with gfile.FastGFile('mmnet.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())
 
output_shape = [1, 256, 256, 2]
img_shape = [1, 256, 256, 3]
output_dir = 'Bin/test/outputs_pb'
img_dir = 'Bin/test/images'
gt_dir = 'Bin/test/masks'
os.makedirs(output_dir, exist_ok=True)

names_images = sorted(os.listdir(img_dir))
for name in names_images:
    filename = os.path.join(img_dir, name)
    mask_filename = os.path.join(gt_dir, name)

    test_img = np.fromfile(filename, dtype=np.float32).reshape([1, 256, 256, 3])
    output = sess.run('output:0', feed_dict={'input_x:0': test_img})
    # test_mask = np.fromfile(mask_filename, dtype=np.float32).reshape([1, 256, 256])

    # output = sess.run('output:0', feed_dict={'input_img:0': test_img, 'input_mask:0': test_mask})
    output.tofile(os.path.join(output_dir, name.split('.')[0] + '_output_0.bin'))