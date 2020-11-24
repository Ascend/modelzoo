# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

""" utilities for hooks """

import tensorflow as tf

from noahnmt.configurable import _create_from_dict
from noahnmt.hooks import wps_hook
from noahnmt.hooks import obs_hook
from noahnmt.utils import graph_utils


def create_hooks_from_dict(hooks_dict, config, hparams=None):
  """ create hooks from dict, mainly train_hooks
  """
  if not hooks_dict:
    return []
    
  hooks = []
  for dict_ in hooks_dict:
    hook = _create_from_dict(
        dict_, 
        model_dir=config.model_dir,
        run_config=config)
    hooks.append(hook)
  return hooks


def create_helpful_hooks(run_config):
  hooks = []
  # wps hook
  hooks.append(
      wps_hook.WpsCounterHook(
          every_n_steps=run_config.log_step_count_steps)
  )
  
  # obs hook
  if run_config.model_dir.startswith("s3://"):
    hooks.append(
        obs_hook.CheckpointSyncToOBSHook(
          run_config.model_dir,
          save_secs=run_config.save_checkpoints_secs,
          save_steps=run_config.save_checkpoints_steps
        )
    )
  return hooks
  