from __future__ import print_function
import tensorflow as tf
import logging
import numpy as np
import time
import sys,os

from absl import app as absl_app
from absl import flags
#from absl import logging
FLAGS = flags.FLAGS

rank_size = int(os.getenv('RANK_SIZE'))

class LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, print_freq):
  #  def __init__(self, global_batch_size, num_records, display_every=10, logger=None):
        self.iter_times = []
        self.display_every = print_freq
        self.log_dir  = './model_ckpt'
        self.logger = get_logger('NCF.log', self.log_dir)
        self.batch_size = FLAGS.batch_size
        self.iterator_per_loop = FLAGS.iterations_per_loop 


    def after_create_session(self, session, coord):
        rank0log(self.logger, 'Step  time_per_step   Loss')
        self.elapsed_secs = 0.
        self.count = 0

    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.train.SessionRunArgs( 
            fetches=[tf.train.get_global_step(), 'cross_entropy:0'])
#                     'loss:0', 'loss:0', 'learning_rate:0'])

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        self.iter_times.append(batch_time)
        self.elapsed_secs += batch_time
        self.count += 1
        global_step, loss= run_values.results
        if global_step == 1 or global_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec =  self.batch_size * self.iterator_per_loop * rank_size / dt
            self.logger.info('step:%i   fps:%7.5f  loss:%6.6f' %
                             (global_step, img_per_sec, loss))
            self.elapsed_secs = 0.
            self.count = 0



def rank0log(logger, *args, **kwargs):
    if logger: 
        logger.info(''.join([str(x) for x in list(args)]))
    else:
        print(*args, **kwargs)


def get_logger(log_name, log_dir):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)  # INFO, ERROR
    # file handler which logs debug messages
    if not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            # if log_dir is common for multiple ranks like on nfs
            pass
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # add formatter to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

