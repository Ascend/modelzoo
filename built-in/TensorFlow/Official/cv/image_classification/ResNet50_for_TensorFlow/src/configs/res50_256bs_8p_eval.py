# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf

import os
log_dir = './resnet50_train/results/'+os.path.basename(__file__).split('.')[0]

#256
config = {
    # ============ for testing =====================
    'accelerator': '1980',    # 'gpu', '1980' 
    'shuffle_enable': 'yes',
    'shuffle_buffer_size': 10000,
    'rank_size': 8, 
    'shard': True,

    # ======= basic config ======= # 
    'mode':'train',                                         # "train","evaluate","train_and_evaluate"
    'epochs_between_evals': 4,                              #used if mode is "train_and_evaluate"
    'stop_threshold': 80.0,                                 #used if mode is "train_and_evaluate"
    'data_dir':'/opt/npu/resnet_data_new',
    'data_url': 'file://PATH_TO_BE_CONFIGURED',
    'data_type': 'TFRECORD',
    'model_name': 'resnet50', 
    'num_classes': 1001,
    'num_epochs': 90,
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
    'batch_size':256,                                        #minibatch size per node, total batchsize = batch_size*hvd.size()*itersize
                               
    'momentum': [0.9],

    #=======  data processing config =======
    'min_object_covered': 0.1,                              #used for random crop
    'aspect_ratio_range':[3. / 4., 4. / 3.],
    'area_range':[0.16, 1.0],
    'max_attempts': 100,

    #=======  data augment config ======= 
    'increased_aug': False,
    'brightness':0.3,
    'saturation': 0.6,
    'contrast': 0.6,
    'hue': 0.13,
    'num_preproc_threads': 22,

    #=======  initialization config ======= 
    'conv_init': tf.variance_scaling_initializer(),
    'bn_init_mode': 'adv_bn_init',                         # "conv_bn_init" or "adv_bn_init",initializer the gamma in bn in different modes
                                                            # "adv_bn_init" means initialize gamma to 0 in each residual block's last bn, and initialize other gamma to 1
                                                            # "conv_bn_init" means initialize all the gamma to a constant, defined by "bn_gamma_initial_value"
    'bn_gamma_initial_value': 1.0,

    #======== model architecture ==========
    'resnet_version': 'v1.5',  
    'arch_type': 'original',                                   # ------ input -------
                                                            # C1,C2,C3: input block, stride in different layer
                                                            # ------ shortcut ------
                                                            # D1: average_pooling + conv1*1 in shortcut  in downsample block
                                                            # D2: conv3*3,stride=2 in shortcut in downsample block
                                                            # D3: conv1*1 +average_pooling in shortcut  in downsample block
                                                            # ------ mainstream ----
                                                            # E1: average_pooling + conv3*3 in mainstream in downsample block  
                                                            # E2: conv3*3 + average_pooling in mainstream in downsample block 

    #=======  logger config ======= 
    'display_every': 1,
    'log_name': 'resnet50.log',
    'log_dir': 'PATH_TO_BE_CONFIGURED',

    #=======  Learning Rate Config ======= 
    'lr_warmup_mode': 'linear',                             # "linear" or "cosine"
    'warmup_lr': 0.0,
    'warmup_epochs': 10,
    'learning_rate_maximum': 0.8,                    

    'lr_decay_mode': 'cosine',                              # "steps", "poly", "poly_cycle", "cosine", "linear_cosine", "linear_twice", "constant" for 1980 only
    'learning_rate_end': 0.00001,

    'decay_steps': '10,20,30',                              #for "steps"
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

def res50_config():
    config['global_batch_size'] = config['batch_size'] * config['rank_size']
    config['do_checkpoint'] = True

    return config
