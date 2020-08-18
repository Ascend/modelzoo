import tensorflow as tf
import horovod.tensorflow as hvd
import os

#256
config = {
    # ============ for testing =====================
    #'accelerator': 'gpu',    # 'gpu' 
    'shuffle_enable': 'yes',
    'shuffle_buffer_size': 10000,
    'rank_size': 8, 
    'shard': False,

    # try horovod
    'mlperf_log': False,

    # ======= basic config ======= # 
    'mode':'train_and_evaluate',                                         # "train","evaluate","train_and_evaluate"
    'epochs_between_evals': 5,                              #used if mode is "train_and_evaluate"
    'stop_threshold': 80.0,                                 #used if mode is "train_and_evaluate"
    'data_url': '/data/slimImagenet',
    'data_type': 'TFRECORD',
    'model_name': 'alexnet',
    'num_classes': 1000,
    'num_epochs': 150,
    'height':224,
    'width':224, 
    'dtype': tf.float32,
    'data_format': 'channels_last',
    'use_nesterov': True,
    'eval_interval': 1,
    'loss_scale': 1024,                                #could be float or string. If float, static loss scaling is applied. 
                                                            #If string, the corresponding automatic loss scaling algorithm is used.
                                                            #Must be one of 'Backoff' of 'LogMax' (case insensitive).
    'use_lars': False,
    'label_smoothing':0.1,                                  #If greater than 0 then smooth the labels.
    'weight_decay': 0.0001,
    'batch_size': 128,                                        #minibatch size per node, total batchsize = batch_size*hvd.size()*itersize
                               
    'momentum': [0.9],

    #=======  data processing config =======
    'min_object_covered': 0.1,                              #used for random crop
    'aspect_ratio_range':[3. / 4., 4. / 3.],
    'area_range':[0.08, 1.0],
    'max_attempts': 100,

    'aug_method': 'hxb',

    #=======  data augment config ======= 
    'increased_aug': False,
    'brightness':0.3,
    'saturation': 0.6,
    'contrast': 0.6,
    'hue': 0.13,
    'num_preproc_threads': 22,


    #======== model architecture ==========
    'alexnet_version': 'he_uniform',
    'arch_type': 'original',                                 

    #=======  logger config ======= 
    'display_every': 500,
    'log_name': 'alexnet_8p.log',
    'log_dir': './results/model_8p', # changed preprocessing according to ME version , lr schedule
   

    #=======  Learning Rate Config ======= 
    'lr_warmup_mode': 'linear',                             # "linear" or "cosine"
    'warmup_lr': 0.0,
    'warmup_epochs': 5,
    'learning_rate_maximum': 0.06,

    'lr_decay_mode': 'cosine',                              # "steps", "poly", "poly_cycle", "cosine", "linear_cosine", "linear_twice", "constant" for 1980 only
    'learning_rate_end': 0.00001,

    'decay_steps': '30,60,90,120',
    'lr_decay_steps': '6.4,0.64,0.064',

    'ploy_power': 2.0,                                      #for "poly" and "poly_cycle"

    'cdr_first_decay_ratio': 0.33,                          #for "cosine_decay_restarts"
    'cdr_t_mul':2.0,
    'cdr_m_mul':0.1,

    'lc_periods':0.47,                                      #for "linear_consine"
    'lc_beta':0.00001, 
    
    'lr_mid': 0.5,                                          #for "linear_twice"
    'epoch_mid': 80,
    
    'bn_lr_scale':1.0,

  }



def configure():
    # add horovod for multiGPU
    hvd.init()
    config['global_batch_size'] = config['batch_size'] * hvd.size()
    config['hvd'] = hvd
    config['do_checkpoint'] = (hvd.rank() == 0)

    return config


