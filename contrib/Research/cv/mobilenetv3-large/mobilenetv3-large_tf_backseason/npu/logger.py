from __future__ import print_function
import tensorflow as tf
import logging
import numpy as np
import time
import sys, os


class LogSessionRunHook(tf.estimator.SessionRunHook):
  def __init__(self, config, warmup_steps=5):
    #  def __init__(self, global_batch_size, num_records, display_every=10, logger=None):
    self.global_batch_size = config['global_batch_size']
    self.iterations_per_loop = config['iterations_per_loop']
    self.warmup_steps = warmup_steps
    self.iter_times = []
    self.num_records = config['num_training_samples']
    self.display_every = config['display_every']
    self.logger = get_logger(config['log_name'], config['log_dir'])
    rank0log(self.logger, 'PY' + str(sys.version) + 'TF' + str(tf.__version__))

  def after_create_session(self, session, coord):
    rank0log(self.logger, 'Step   Epoch   Speed   Loss   FinLoss   LR')
    self.elapsed_secs = 0.
    self.count = 0

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.estimator.SessionRunArgs(
      fetches=[tf.compat.v1.train.get_global_step(), 'loss:0', 'total_loss:0', 'learning_rate:0'])

  def after_run(self, run_context, run_values):
    batch_time = time.time() - self.t0
    self.iter_times.append(batch_time)
    self.elapsed_secs += batch_time
    self.count += 1
    global_step, loss, total_loss, lr = run_values.results
    if global_step == 1 or global_step % self.display_every == 0:
      dt = self.elapsed_secs / self.count
      img_per_sec = self.global_batch_size * self.iterations_per_loop / dt
      epoch = global_step * self.global_batch_size / self.num_records
      self.logger.info(f'step:{global_step}  epoch:{epoch} ips:{img_per_sec} '
                       f'loss:{loss}  total_loss:{total_loss}  lr:{lr}, ')
      self.elapsed_secs = 0.
      self.count = 0

  def get_average_speed(self):
    avg_time = np.mean(self.iter_times[self.warmup_steps:])
    speed = self.global_batch_size / avg_time
    return speed


def rank0log(logger, *args, **kwargs):
#  if get_hvd_rank() == 0:
#    if logger:
#      logger.info(''.join([str(x) for x in list(args)]))
#    else:
#      print(*args, **kwargs)
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
#  if not get_hvd_rank():
#    fh = logging.FileHandler(os.path.join(log_dir, log_name))
#    fh.setLevel(logging.DEBUG)
#    fh.setFormatter(formatter)
#    # add handlers to logger
#    logger.addHandler(fh)
  fh = logging.FileHandler(os.path.join(log_dir, log_name))
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  # add handlers to logger
  logger.addHandler(fh)
  return logger
