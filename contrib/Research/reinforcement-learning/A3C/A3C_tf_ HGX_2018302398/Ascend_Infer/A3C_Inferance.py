import sys
import os

import acl

from utils import *
from acl_model import Model
from PIL import Image, ImageDraw, ImageFont
import argparse
import numpy as np
import gym
import cv2
import time
from envs import create_atari_env


class A3C(object):
    def __init__(self, model_path, model_width, model_height):
        self.device_id = 0
        self.context = None
        self.stream = None
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        
    def __del__(self):
        if self._model:
            del self._model
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
    
    def _init_resource(self):
        ret = acl.init()
        check_ret("acl.rt.set_device", ret)

        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        check_ret("acl.rt.get_run_mode", ret)
        
    def init(self):
        #初始化 acl 资源
        self._init_resource() 

        #加载模型
        self._model = Model(self.run_mode, self._model_path)
        ret = self._model.init_resource()
        if ret != SUCCESS:
            return FAILED

        return SUCCESS
    
    def inference(self, image):
        return self._model.execute(image)
        

def parse_args():
    """get parameters from commands"""
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_model', default='../om_model/a3c_pong_model.om',
                        help="""path of input model""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    
    return args
    

def main():
    args = parse_args()
    
    MODEL_PATH = args.input_model
    MODEL_WIDTH=42
    MODEL_HEIGHT=42
    
    #T0 = time.time()
    print(MODEL_PATH)
    #模型实例化
    model = A3C(MODEL_PATH, MODEL_WIDTH, MODEL_HEIGHT)
    
    #推理初始化
    ret = model.init()
    check_ret("Classify.init ", ret)
    
    T0 = time.time()    
    #交互器初始化
    env = env = create_atari_env('PongDeterministic-v4')
    env_Shape = env.observation_space.shape
    env_Dim = len(env_Shape)
    N_A = env.action_space.n
    num = 10

    for i in range(num):
        env.seed(2) 
        s = env.reset()
        score = 0
        frame = 0
        t0 = time.time()

        while True:
            s = s.astype(np.float32)
            s = s.squeeze()[np.newaxis,:,:,np.newaxis]
            a = model.inference(s)
            a = int(a[0])
            s_, r, done, info = env.step(a)
            score += r
            frame += 1
            s = s_
            if done:
                break
        print("Score: %d; Frames: %d; time: %f"%(score, frame, time.time()-t0))
    #print("[Infer Finished] Frames: %d ; Score: %d ; Time: %f"%(frame, score, T1-T0))
    
if __name__ == '__main__':
    main()
