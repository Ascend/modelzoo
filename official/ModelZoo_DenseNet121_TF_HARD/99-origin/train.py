import tensorflow as tf
import numpy as np

import sys
import ast

import densenet.data_loader as dl
import densenet.host_model as ml
import densenet.hyper_param as hp
import densenet.layers as ly
import densenet.logger as lg
import densenet.trainer as tr
import densenet.create_session as cs

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
    #cmdline.add_argument('--log_name', default="log.log",
    #                     help="""log file name.""")
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    cfg_file = FLAGS.config_file
    cfg = __import__(cfg_file)

    config = cfg.densenet_config()
    config['iterations_per_loop'] = int(FLAGS.iterations_per_loop)
    config['max_train_steps'] = int(FLAGS.max_train_steps)
    config['debug'] = FLAGS.debug
    config['eval'] = FLAGS.eval
    config['model_dir'] = FLAGS.model_dir
    #config['log_name'] = FLAGS.log_name
    print("iterations_per_loop:%d" %(config['iterations_per_loop']))
    print("max_train_steps    :%d" %(config['max_train_steps']))
    print("debug              :%s" %(config['debug']))
    print("eval               :%s" %(config['eval']))
    print("model_dir          :%s" %(config['model_dir']))

    Session = cs.CreateSession(config)
    data = dl.DataLoader(config)
    hyper_param = hp.HyperParams(config)
    layers = ly.Layers()
    logger = lg.LogSessionRunHook(config)
    model = ml.Model(config, data, hyper_param, layers, logger)
   
    trainer = tr.Trainer(Session, config, data, model, logger)

    #trainer.train()
    trainer.train_and_evaluate()


if __name__ == '__main__':
    main()

