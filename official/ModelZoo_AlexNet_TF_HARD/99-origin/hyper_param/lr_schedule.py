import tensorflow as tf
import numpy as np

import math

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

def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi*last_epoch / T_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)



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

def get_vgg_lr(config, global_step, lr_init, lr_end, lr_max, warmup_epohs, steps_per_epoch, nsteps, lr_decay_mode):
    lr_each_step = []

    if lr_decay_mode == 'steps':
        # decay parameter from gluoncv
        # we directly use the given learning rates
        decay_epoch_index = [50 * steps_per_epoch, 80 * steps_per_epoch]
        total_steps = int(nsteps)
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = 0.01
            elif i < decay_epoch_index[1]:
                lr = 0.001
            else:
                lr = 0.0001
            lr_each_step.append(lr)
    else:
        print("Wrong learning rate parameter.")

    # current_step = tf.to_int32( tf.cast(global_step,tf.float32) / float(steps_per_epoch) )
    current_step = global_step
    lr_each_step = tf.convert_to_tensor( lr_each_step )
    print (lr_each_step)
    learning_rate = tf.gather( lr_each_step, current_step )

    return learning_rate

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
          if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i) 
          elif i <= total_steps:
            base =  ( 1.0 - (float(i)-float(warmup_steps))/(float(total_steps)-float(warmup_steps)) ) 
            lr = float(lr_max) * base 
          else:
            lr = 0.0
          lr_each_step.append(lr)

    elif lr_decay_mode == 'cosine':
        total_steps = int(nsteps)
        
        warmup_steps = steps_per_epoch * warmup_epochs
        for i in range( config['total_steps_include_iterations'] ):
          if i < warmup_steps:
            lr = cos_warmup_1980( i, warmup_steps, lr_max )
          elif i <= total_steps:
            lr = cos_decay_1980( i, warmup_steps, total_steps, lr_max )
          else:
            lr = 0.0
          lr_each_step.append(lr)
    elif lr_decay_mode == 'linear_cosine':
        total_steps = int(nsteps)
        warmup_steps = steps_per_epoch * warmup_epochs
        inc_each_step = ( float(lr_max) - float(lr_init) ) / float(warmup_steps)
        for i in range( config['total_steps_include_iterations'] ):
          if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i) 
          elif i <= total_steps:
            lr = cos_decay_1980( i, warmup_steps, total_steps, lr_max )
          else:
            lr = 0.0
          lr_each_step.append(lr)
    else:
        total_steps = int(nsteps)
        warmup_steps = steps_per_epoch * warmup_epochs
        for i in range(total_steps):
            if i < warmup_steps:
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



