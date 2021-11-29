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
import numpy as np

def get_lr(lr, lr_end, lr_decay_mode, warmup_it, decay_steps, global_step, steps, lr_steps, ploy_power,
           cdr_first_decay_ratio, cdr_t_mul, cdr_m_mul, cdr_alpha, cd_alpha, lc_periods, lc_alpha, lc_beta, lr_mid, it_mid):
    if lr_decay_mode == 'steps':
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    steps, lr_steps)
    elif lr_decay_mode == 'poly' or lr_decay_mode == 'poly_cycle':
        cycle = lr_decay_mode == 'poly_cycle'
        learning_rate = tf.train.polynomial_decay(lr,
                                                  global_step - warmup_it,
                                                  decay_steps=decay_steps - warmup_it,
                                                  end_learning_rate=lr_end,
                                                  power=ploy_power,
                                                  cycle=cycle)
    elif lr_decay_mode == 'cosine_decay_restarts':
        learning_rate = tf.train.cosine_decay_restarts(lr, 
                                                       global_step - warmup_it,
                                                       (decay_steps - warmup_it) * cdr_first_decay_ratio,
                                                       t_mul=cdr_t_mul, 
                                                       m_mul=cdr_m_mul,
                                                       alpha=cdr_alpha)
    elif lr_decay_mode == 'cosine':
        learning_rate = tf.train.cosine_decay(lr,
                                              global_step - warmup_it,
                                              decay_steps=decay_steps - warmup_it,
                                              alpha=cd_alpha) 
    elif lr_decay_mode == 'linear_cosine':
        learning_rate = tf.train.linear_cosine_decay(lr,
                                                     global_step - warmup_it,
                                                     decay_steps=decay_steps - warmup_it,
                                                     num_periods=lc_periods,#0.47,
                                                     alpha=lc_alpha,#0.0,
                                                     beta=lc_beta)#0.00001)
    elif lr_decay_mode == 'linear_twice':
        learning_rate = decay_linear_twice(lr, lr_mid, lr_end, warmup_it, it_mid, decay_steps, global_step )

    else:
        raise ValueError('Invalid type of lr_decay_mode')
    return learning_rate


def cos_warmup_1980(  global_step, warmup_steps, max_lr ):
    PI = 3.14159265359
    ang = PI +  PI * ( float(global_step+1) / float(warmup_steps) )
    offset  = max_lr * 0.5*( 1.0 + np.cos( ang ) )
    return offset

def cos_decay_1980(  global_step, warmup_steps, total_steps, max_lr ):
    PI = 3.14159265359
    ang =  PI * ( float(global_step - warmup_steps+1) / float(total_steps - warmup_steps) )
    offset  = max_lr * 0.5*( 1.0 + np.cos( ang ) )
    return offset


def get_1980_lr(config, global_step, lr_init, lr_end, lr_max, warmup_epochs, steps_per_epoch, nsteps, lr_decay_mode):
    lr_each_step = []

    if lr_decay_mode == 'steps':
        decay_epoch_index = [30 * steps_per_epoch,60 * steps_per_epoch,80 * steps_per_epoch]
        total_steps = int(nsteps)
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)
    elif lr_decay_mode == 'poly':
        total_steps = int(nsteps)
        warmup_steps = steps_per_epoch * warmup_epochs
        inc_each_step = ( float(lr_max) - float(lr_init) ) / float(warmup_steps)
        for i in range( config['total_steps_include_iterations'] ):
          if i <= warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i) 
          elif i < total_steps:
            base =  ( 1.0 - (float(i)-float(warmup_steps))/(float(total_steps)-float(warmup_steps)) ) 
            lr = float(lr_max) * base 
          else:
            lr = 0.0
          lr_each_step.append(lr)

    elif lr_decay_mode == 'cosine':
        total_steps = int(nsteps)
        
        warmup_steps = steps_per_epoch * warmup_epochs
        for i in range( config['total_steps_include_iterations'] ):
          if i <= warmup_steps:
            lr = cos_warmup_1980( i, warmup_steps, lr_max )
          elif i < total_steps:
            lr = cos_decay_1980( i, warmup_steps, total_steps, lr_max )
          else:
            lr = 0.0
          lr_each_step.append(lr)
    elif lr_decay_mode == 'linear_cosine':
        total_steps = int(nsteps)
        warmup_steps = steps_per_epoch * warmup_epochs
        inc_each_step = ( float(lr_max) - float(lr_init) ) / float(warmup_steps)
        for i in range( config['total_steps_include_iterations'] ):
          if i <= warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i) 
          elif i < total_steps:
            lr = cos_decay_1980( i, warmup_steps, total_steps, lr_max )
          else:
            lr = 0.0
          lr_each_step.append(lr)
    else:
        total_steps = int(nsteps)
        warmup_steps = steps_per_epoch * warmup_epochs
        for i in range(total_steps):
            if i <= warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else: 
                lr = lr_max - ( lr_max - lr_end ) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append( lr )

   # current_step = tf.to_int32( tf.cast(global_step,tf.float32) / float(steps_per_epoch) )
    current_step = global_step
    lr_each_step = tf.convert_to_tensor( lr_each_step )
    print (lr_each_step)
    learning_rate = tf.gather( lr_each_step, current_step )

    return learning_rate

def warmup_decay(lr_warmup_mode, warmup_lr, global_step, warmup_steps, warmup_end_lr):
    if lr_warmup_mode == 'linear':
        learning_rate = linear_warmup(warmup_lr, global_step, warmup_steps, warmup_end_lr)
    elif lr_warmup_mode == 'cosine':
        learning_rate = cos_warmup(warmup_lr, global_step, warmup_steps, warmup_end_lr)
    else:
        raise ValueError('Invalid type of lr_warmup_mode')
    return learning_rate


def linear_warmup(warmup_lr, global_step, warmup_steps, warmup_end_lr):
    from tensorflow.python.ops import math_ops
    p = tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    diff = math_ops.subtract(warmup_end_lr, warmup_lr)
    res = math_ops.add(warmup_lr, math_ops.multiply(diff, p))
    return res

def cos_warmup( warmup_lr, global_step, warmup_steps, warmup_end_lr ):
    PI = 3.14159265359
    diff = tf.subtract( warmup_end_lr, warmup_lr )
    ang = PI +  PI * ( tf.cast( global_step, tf.float32 ) / tf.cast( warmup_steps,tf.float32 ))
    offset = diff * 0.5 * ( 1.0 + tf.math.cos( ang ) )
    res =  tf.add( warmup_lr, offset )
    return res


def decay_linear( lr_start, lr_end, it_start, it_end, global_step ):
    down_steps = it_end - it_start
    down_range = lr_start - lr_end 
    down_per_step = float( down_range ) / float( down_steps )
    res = tf.subtract( tf.cast(lr_start, tf.float32),  tf.multiply( tf.cast(down_per_step, tf.float32), tf.subtract(tf.cast(global_step, tf.float32), tf.cast(it_start, tf.float32) )) )
    return res

def decay_linear_twice(lr_start, lr_mid, lr_end, it_start, it_mid, it_end, global_step ):
    learning_rate = tf.cond( global_step < it_start, lambda: tf.cast(lr_start, tf.float32), lambda: decay_linear(lr_start, lr_mid, it_start, it_mid, global_step))
    learning_rate = tf.cond( global_step > it_mid, lambda: decay_linear(lr_mid, lr_end, it_mid, it_end, global_step) , lambda: learning_rate )
    return learning_rate



