import tensorflow as tf
import numpy as np

import sys
import ast

import vgg16.data_loader as dl
import vgg16.model as ml
import vgg16.hyper_param as hp
import vgg16.layers as ly
import vgg16.logger as lg
import vgg16.trainer as tr
import vgg16.create_session as cs

import argparse


def main():
    #-------------------choose the config file in .sh file-----------
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--config_file', default="",
                         help="""config file used.""")
    cmdline.add_argument('--model_dir', default="./model_dir",
                         help="""config file used.""")
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    cfg_file = FLAGS.config_file
    cfg = __import__(cfg_file)

    config = cfg.vgg16_config()
    config['model_dir'] = FLAGS.model_dir
    print("model_dir          :%s" %(config['model_dir']))

    Session = cs.CreateSession(config)
    data = dl.DataLoader(config)    
    hyper_param = hp.HyperParams(config)
    layers = ly.Layers() 
    logger = lg.LogSessionRunHook(config)
    model = ml.Model(config, data, hyper_param, layers, logger)
   
    trainer = tr.Trainer(Session, config, data, model, logger)

    if config['mode'] == 'train':
        trainer.train()
    elif config['mode'] == 'evaluate':
        trainer.evaluate()
    elif config['mode'] == 'train_and_evaluate':
        trainer.train_and_evaluate()
    else:
        raise ValueError("Invalid mode.")


if __name__ == '__main__':
    main()

