# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

try:
  LOCAL_CACHE_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
  import moxing as mox
except KeyError:
  tf.logging.info("Local machine mode")

class CheckpointSyncToOBSHook(tf.train.CheckpointSaverHook):
  def __init__(self,
               checkpoint_dir,
               save_secs=None,
               save_steps=None,
               saver=None,
               checkpoint_basename="model.ckpt",
               scaffold=None,
               listeners=None):
    self._s3_ckpt_dir = checkpoint_dir
    self._ckpt_basename = checkpoint_basename
    self._saved_graph = False
    checkpoint_dir = os.path.join(LOCAL_CACHE_DIR, "model_dir")
    super(CheckpointSyncToOBSHook, self).__init__(
               checkpoint_dir=checkpoint_dir,
               save_secs=save_secs,
               save_steps=save_steps,
               saver=saver,
               checkpoint_basename=checkpoint_basename,
               scaffold=scaffold,
               listeners=listeners)

  def _save(self, session, step):
    # save to local dir
    super(CheckpointSyncToOBSHook, self)._save(session, step)
    # find the lastest ckpt files
    log_files_list = mox.file.list_directory(self._checkpoint_dir)
    ckpt_files_list = []
    cur_ckpt_file_prefix = '%s-%d' % (self._save_path, step)
    for log_file in log_files_list:
      abs_log_file = os.path.join(self._checkpoint_dir, log_file)
      if abs_log_file.startswith(cur_ckpt_file_prefix):
        ckpt_files_list.append(log_file)
      elif not log_file.startswith(self._ckpt_basename):
        ckpt_files_list.append(log_file)
    #ckpt_files_list.append("checkpoint")
    # log and copy
    tf.logging.info('Uploading checkpoints from local to OBS...')
    tf.logging.info(ckpt_files_list)
    mox.file.copy_parallel(self._checkpoint_dir, self._s3_ckpt_dir, file_list=ckpt_files_list)