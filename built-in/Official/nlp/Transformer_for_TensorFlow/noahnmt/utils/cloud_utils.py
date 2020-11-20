# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

""" Utilities for using on Huawei Cloud
"""

import os
import logging

import tensorflow as tf

try:
  LOCAL_CACHE_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
  import moxing as mox
except KeyError:
  tf.logging.info("Local machine mode")


def set_log_file(log_file):
  """ save log into file
  """
  logFormatter = logging.Formatter(logging.BASIC_FORMAT, None)
  rootLogger = logging.getLogger('')

  fileHandler = logging.StreamHandler(tf.gfile.GFile(log_file, 'w'))
  fileHandler.setLevel(logging.DEBUG)
  fileHandler.setFormatter(logFormatter)
  rootLogger.addHandler(fileHandler)