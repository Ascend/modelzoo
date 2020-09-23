import tensorflow as tf
import math
import numpy as np

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

class HyperParams:
    def __init__(self, config):
        self.config=config
        nsteps_per_epoch = self.config['num_training_samples'] // self.config['global_batch_size']
        self.config['nsteps_per_epoch'] = nsteps_per_epoch
        # nstep = self.config['num_training_samples'] * self.config['num_epochs'] // self.config['global_batch_size']
        if self.config['num_epochs']:
            nstep = nsteps_per_epoch * self.config['num_epochs']   #------calculate nsteps in a different way------
        else:
            nstep = self.config['max_train_steps']
        self.config['nstep'] = nstep
        
        #self.config['total_steps_include_iterations'] = int( self.config['nstep'] + self.config['iterations_per_loop'])
        self.config['save_summary_steps'] = nsteps_per_epoch
        self.config['save_checkpoints_steps'] = nsteps_per_epoch

        self.cos_lr = warmup_cosine_annealing_lr(0.1, nsteps_per_epoch, 0, 150, 150, 0.)

    def get_hyper_params(self):
        hyper_params = {}
        hyper_params['learning_rate'] = self.get_learning_rate()

        return hyper_params

    # cos_lr
    def get_learning_rate(self):
        global_step = tf.train.get_global_step()
    
        learning_rate = tf.gather(tf.convert_to_tensor(self.cos_lr), global_step)

        learning_rate = tf.identity(learning_rate, 'learning_rate')

        return learning_rate




def get_densenet121_lr(global_step, steps_per_epoch, nsteps):
    lr_each_step = []

    decay_epoch_index = [40 * steps_per_epoch, 80 * steps_per_epoch]
    total_steps = int(nsteps)
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = 0.1
        elif i < decay_epoch_index[1]:
            lr = 0.01
        else:
            lr = 0.001
        lr_each_step.append(lr)
    # else:
    #    print("Wrong learning rate parameter.")

    # current_step = tf.to_int32( tf.cast(global_step,tf.float32) / float(steps_per_epoch) )
    current_step = global_step
    lr_each_step = tf.convert_to_tensor(lr_each_step)
    # print (lr_each_step)
    learning_rate = tf.gather(lr_each_step, current_step)

    return learning_rate

