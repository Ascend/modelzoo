import tensorflow as tf
import horovod.tensorflow as hvd
import os

config = {
    # ============ for testing =====================
    'accelerator': 'gpu',    # 'gpu', 'npu' 
#    'shuffle_enable': 'yes',
    'shuffle_buffer_size': 10000,
    'rank_size': 8, 
    'shard': False,

    # ======= basic config ======= # 
    'mode':'train_and_evaluate',                                         # "train","evaluate","train_and_evaluate"
    #'mode': 'train',
    'epochs_between_evals': 1,                              #used if mode is "train_and_evaluate"
    'data_url': '/workspace/tf_npu/slimImagenet',
    #'data_url': '/opt/npu/z00438116/tf/vgg16_train/mains/slimImagenet',
    #'data_url': '/raid5/dataset/slimImagenet',
    'num_epochs': 120,
    'height':224,
    'width':224, 
    'dtype': tf.float32,
#    'data_format': 'channels_last',
    'use_nesterov': True,
#    'eval_interval': 1,
#    'loss_scale': 1024,                                #could be float or string. If float, static loss scaling is applied. 
                                                            #If string, the corresponding automatic loss scaling algorithm is used.
                                                            #Must be one of 'Backoff' of 'LogMax' (case insensitive).
    'use_lars': False,
    'label_smoothing':0.1,                                  #If greater than 0 then smooth the labels.
    'weight_decay': 0.0001,
    'batch_size': 32,                                        #minibatch size per node, total batchsize = batch_size*hvd.size()*itersize
                               
    'momentum': [0.9],

    #=======  data processing config =======
#    'min_object_covered': 0.1,                              #used for random crop
#    'aspect_ratio_range':[3. / 4., 4. / 3.],
#    'area_range':[0.16, 1.0],
#    'max_attempts': 100,

    'aug_method': 'me',

    #=======  data augment config ======= 
    'increased_aug': False,
    'brightness':0.3,
    'saturation': 0.6,
    'contrast': 0.6,
    'hue': 0.13,
    'num_preproc_threads': 22,

    #=======  logger config ======= 
    'display_every': 1251,
    'log_name': 'densenet_8p_host.log',
    'log_dir': './model_dir_8p_xla_amp',

    #=======  Learning Rate Config ======= 
#    'lr_warmup_mode': 'linear',                             # "linear" or "cosine"
#    'warmup_lr': 0.0,
#    'warmup_epochs': 10,
    #'learning_rate_maximum': 0.1,                    
#    'learning_rate_maximum': 0.01,

#    'lr_decay_mode': 'constant',
    'lr_decay_mode': 'cosine',                              # "steps", "poly", "poly_cycle", "cosine", "linear_cosine", "linear_twice", "constant" for 1980 only
#    'learning_rate_end': 0.00001,


  }


def densenet_config():
    # add horovod for multiGPU
    hvd.init()
    #config['global_batch_size'] = config['batch_size'] * hvd.size()
    #config['global_batch_size'] = config['batch_size'] * get_rank_size()
    config['global_batch_size'] = config['batch_size'] * config['rank_size']
    #config['do_checkpoint'] = True
    config['do_checkpoint'] = (hvd.rank() == 0)
    #config['do_checkpoint'] = get_rank_id() == 0
    #config['do_checkpoint'] = (int(os.getenv('DEVICE_ID')) == 0)

    return config

