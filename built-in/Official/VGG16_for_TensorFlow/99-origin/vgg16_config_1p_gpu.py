import tensorflow as tf
import os
import horovod.tensorflow as hvd

config = {
    'rank_size': 1, 
    'shard': False,

    # ======= basic config ======= # 
    'mode':'train',                                         # "train","evaluate","train_and_evaluate"

    # modify here for train_and_evaluate mode
    'epochs_between_evals': 1,                              #used if mode is "train_and_evaluate"
    'num_epochs': None,
    
    # modify here for train mode
    'max_train_steps': 100,

    'data_url': '/data/slimImagenet',
    'dtype': tf.float32,
    'use_nesterov': True,
    'label_smoothing':0.1,                                  #If greater than 0 then smooth the labels.
    'weight_decay': 0.0001,
    'batch_size': 32,                                        #minibatch size per node, total batchsize = batch_size*hvd.size()*itersize
                               
    'momentum': [0.9],

    'lr': 0.001,
    'max_epoch': 150,

    #=======  logger config ======= 
    'display_every': 10,
    'log_name': 'vgg16.log',
    'log_dir': 'ckpt',
  }


def vgg16_config():
    hvd.init()
    config['global_batch_size'] = config['batch_size'] * config['rank_size']
    config['do_checkpoint'] = (hvd.rank() == 0)

    return config

