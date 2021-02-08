import tensorflow as tf
import numpy as np
import gym
from envs import create_atari_env
from tensorflow.python.platform import gfile
import os
import time
import argparse

def parse_args():
    """get parameters from commands"""
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_name', default='AlienDeterministic-v4',
                        help="""name of game""")
    parser.add_argument('--model_path', default='./pb_model/a3c_pong_model',
                        help="""model path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    
    return args

class pbInference(object):
    def __init__(self, PATH_TO_CKPT, WIDTH_HEIGH):
        pid = os.getpid()
        print('------init pid-----', pid)
        self.WIDTH_HEIGH = WIDTH_HEIGH
        self.sess = tf.Session
        #self.sess = tf.Session(graph=g)
        with gfile.FastGFile(PATH_TO_CKPT, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
        self.sess.run(tf.global_variables_initializer())
 
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
 
    def service(self, job_id, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_np = self.load_image_into_numpy_array(image)
        image_np = cv2.resize(image_np, (self.WIDTH_HEIGH, self.WIDTH_HEIGH))
        image_np_expanded = np.expand_dims(image_np, axis=0)
 
        input = self.sess.graph.get_tensor_by_name("input:0")
        label = self.sess.graph.get_tensor_by_name("output:0")
        output_dict = self.sess.run(label, feed_dict={input: image_np_expanded})
        print('output_dict:',output_dict)
        label = np.array(output_dict[0]).argmax()
        score = output_dict[0][label]
        return label,score
        
args = parse_args()
game_Name = args.game_name
model_path = args.model_path
        
sessionConfig = tf.ConfigProto()
sessionConfig.gpu_options.allow_growth = True
sess = tf.Session(config=sessionConfig)

with gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图
sess.run(tf.global_variables_initializer())

input = sess.graph.get_tensor_by_name("s:0")
a_out = sess.graph.get_tensor_by_name("out_a:0")

T0 = time.time()
env = create_atari_env(game_Name)

for seed in range(10):
    score = 0
    frame = 0
    lstm_state = []
    c_state = np.zeros((1, 256), np.float32)
    h_state = np.zeros((1, 256), np.float32)
    
    env.seed(seed)
    s = env.reset()
    t0 = T0
    while True:
        s = np.squeeze(s)[np.newaxis, :, :, np.newaxis]
        #a, c_state, h_state= sess.run([ap_out, c_out, h_out], feed_dict={input:s, c_in:c_state, h_in:h_state})
        a = sess.run(a_out, feed_dict={input:s})
        a = int(a[0])
        s_, r, done, info = env.step(a)
        score += r
        frame += 1
        s = s_
        if done:
            break
    t = time.time()-t0
    t0 = time.time()
    print([score, t])

print((time.time()-T0)/10)