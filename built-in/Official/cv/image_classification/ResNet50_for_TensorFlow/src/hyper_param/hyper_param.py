import tensorflow as tf
from .lr_schedule import warmup_decay, get_lr, get_1980_lr


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
        
        self.config['total_steps_include_iterations'] = int( self.config['nstep'] + self.config['iterations_per_loop'])
        self.config['save_summary_steps'] = nsteps_per_epoch
        self.config['save_checkpoints_steps'] = nsteps_per_epoch


    def get_hyper_params(self):
        hyper_params = {}
        hyper_params['learning_rate'] = self.get_learning_rate()

        return hyper_params
 

    def get_learning_rate(self): 
        global_step = tf.train.get_global_step()
        nsteps_per_epoch = self.config['nsteps_per_epoch']

        warmup_lr = self.config['warmup_lr']
        lr = self.config['learning_rate_maximum']
        lr_end = self.config['learning_rate_end']
        lr_decay_mode = self.config['lr_decay_mode']


       
        with tf.device('/cpu:0'):  # Allow fallback to CPU if no GPU support for these ops

            if lr_decay_mode == 'constant' or self.config['num_epochs'] == None:
                learning_rate = tf.constant(lr, tf.float32)
            else:
                learning_rate = get_1980_lr(self.config, global_step, warmup_lr, lr_end, lr, self.config['warmup_epochs'], nsteps_per_epoch, self.config['nstep'], lr_decay_mode )

            learning_rate = tf.identity(learning_rate, 'learning_rate')
        return learning_rate


