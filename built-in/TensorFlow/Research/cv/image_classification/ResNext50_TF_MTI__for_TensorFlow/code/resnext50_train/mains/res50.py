import tensorflow as tf
import sys
import ast
#sys.path.append("..")
#sys.path.append("../models")
#sys.path.append("./resnet50_train/")
#sys.path.append("./resnet50_train/models")
import os
base_path=os.path.split(os.path.realpath(__file__))[0]
print ("#########base_path:", base_path)
path_1 = base_path + "/.."
print (path_1)
path_2 = base_path + "/../models"
print (path_2)
path_3 = base_path + "/../../"
print (path_3)


sys.path.append(base_path + "/..")
sys.path.append(base_path + "/../models")
sys.path.append(base_path + "/../../")
sys.path.append(base_path + "/../../models")

from utils import create_session as cs
from utils import logger as lg
from data_loader.resnet50 import data_loader as dl
from models.resnet50 import res50_model as ml
from optimizers import optimizer as op
from losses import res50_loss as ls
from trainers import gpu_base_trainer as tr
# from configs import res50_config as cfg
from hyper_param import hyper_param as hp
from layers import layers as ly

import argparse

def main():
    #-------------------choose the config file in .sh file-----------
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--config_file', default="",
                         help="""config file used.""")
    cmdline.add_argument('--iterations_per_loop', default=1,
                         help="""config file used.""")
    cmdline.add_argument('--max_train_steps', default=200,
                         help="""config file used.""")
    cmdline.add_argument('--debug', default=True, type=ast.literal_eval,
                         help="""config file used.""")
    cmdline.add_argument('--eval', default=False, type=ast.literal_eval,
                         help="""config file used.""")
    cmdline.add_argument('--model_dir', default="./model_dir",
                         help="""config file used.""")
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    cfg_file = FLAGS.config_file
    configs = 'configs'
    cfg = getattr(__import__(configs, fromlist=[cfg_file]), cfg_file)
    #------------------------------------------------------------------

    config = cfg.res50_config()
    config['iterations_per_loop'] = int(FLAGS.iterations_per_loop)
    config['max_train_steps'] = int(FLAGS.max_train_steps)
    config['debug'] = FLAGS.debug
    config['eval'] = FLAGS.eval
    config['model_dir'] = FLAGS.model_dir
    print("iterations_per_loop:%d" %(config['iterations_per_loop']))
    print("max_train_steps    :%d" %(config['max_train_steps']))
    print("debug              :%s" %(config['debug']))
    print("eval               :%s" %(config['eval']))
    print("model_dir          :%s" %(config['model_dir']))
    Session = cs.CreateSession(config)
    data = dl.DataLoader(config)
    hyper_param = hp.HyperParams(config)
    layers = ly.Layers() 
    optimizer = op.Optimizer(config)
    loss = ls.Loss(config)
    logger = lg.LogSessionRunHook(config)   # add tensorboard summary

    model = ml.Model(config, data, hyper_param,layers, optimizer, loss, logger)   # get the model 
    trainer = tr.GPUBaseTrain(Session, config, data, model, logger)   # use Estimator to build training process

    if config['mode'] =='train':  
        trainer.train()
        if config['eval'] :
            trainer.evaluate()
    elif config['mode'] =='evaluate':
        trainer.evaluate()
    elif config['mode'] =='train_and_evaluate':
        trainer.train_and_evaluate()
    else:
        raise ValueError('Invalid type of mode')

if __name__ == '__main__':
    main()
