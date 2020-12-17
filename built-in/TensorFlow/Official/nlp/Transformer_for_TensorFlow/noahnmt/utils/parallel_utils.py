# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for using multi-gpus.

From Tensor2Tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import function

from noahnmt.layers import common_layers as common_utils

DEFAULT_DEV_STRING = "existing_device"


def _add_variable_proxy_methods(var, proxy_tensor):
  """Proxy methods of underlying variable.

  This enables our custom getters to still work with, e.g., batch norm.

  Args:
    var: Variable to proxy
    proxy_tensor: Tensor that is identity of var
  """
  proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
  proxy_tensor.assign_sub = var.assign_sub


def transpose_list_of_lists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, "cannot pass the empty list"
  return [list(x) for x in zip(*lol)]


class GPUParamServerDeviceSetter(object):
  def __init__(self, worker_device, ps_devices):
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ["Variable", "VariableV2", "VarHandleOp"]:
      return self.worker_device

    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes),
                          key=lambda x: x[1])
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name


def data_parallelism(daisy_chain_variables=True,
                     all_workers=False,
                     ps_replicas=0,
                     ps_job="/job:ps",
                     ps_gpu=0,
                     schedule="continuous_train_and_eval",
                     sync=False,
                     worker_gpu=1,
                     worker_replicas=1,
                     worker_id=0,
                     gpu_order="",
                     locally_shard_to_cpu=False,
                     worker_job="/job:localhost",
                     no_data_parallelism=False,
                     dp_param_shard=True):
  """See data_parallelism_from_flags."""
  tf.logging.info("schedule=%s" % schedule)
  tf.logging.info("worker_gpu=%s" % worker_gpu)
  tf.logging.info("sync=%s" % sync)
  def _ps_replicas(all_workers=False):
    if all_workers:
      return list(range(ps_replicas))
    # Worker K will be using replicas {0,...n-1} + K*n if we have n replicas.
    num_replicas = ps_replicas // worker_replicas
    return [d + worker_id * num_replicas for d in range(num_replicas)]

  def _gpu_order(num_gpus):
    if gpu_order:
      ret = [int(s) for s in gpu_order.split(" ")]
      if len(ret) == num_gpus:
        return ret
    return list(range(num_gpus))

  def _ps_gpus(all_workers=False):
    ps_gpus = []
    for d in _ps_replicas(all_workers=all_workers):
      ps_gpus.extend([(d, gpu) for gpu in _gpu_order(ps_gpu)])
    return ps_gpus

  def ps_devices(all_workers=False):
    """List of ps devices (where to put the experts).

    Args:
      all_workers: whether the list is for all async workers or just this one.

    Returns:
      a list of device names
    """
    if ps_replicas > 0:
      if ps_gpu > 0:
        return [
            ps_job + "/task:%d/GPU:%d" % (d, gpu)
            for (d, gpu) in _ps_gpus(all_workers=all_workers)
        ]
      else:
        return [
            ps_job + "/task:%d" % d
            for d in _ps_replicas(all_workers=all_workers)
        ]
    else:
      if worker_gpu > 0:
        return ["/GPU:%d" % d for d in _gpu_order(worker_gpu)]
      else:
        return [""]

  def _replica_device_setter(worker_device):
    if ps_replicas == 0:
      return worker_device
    return tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=ps_replicas,
        ps_device=ps_job + "/GPU:0" if ps_gpu > 0 else ps_job)

  is_single_machine = ps_replicas == 0 and worker_replicas == 1

  if no_data_parallelism:
    datashard_devices = [""]
    caching_devices = None
  elif is_single_machine:
    tf.logging.warn(
        "Schedule=%s. Assuming that training is running on a single machine.",
        schedule)
    datashard_devices = ["/GPU:%d" % d for d in _gpu_order(worker_gpu)]
    if locally_shard_to_cpu or worker_gpu < 1:
      datashard_devices += ["/CPU:0"]
    elif dp_param_shard:
      # params are sharded to all devices
      datashard_devices = [
          GPUParamServerDeviceSetter(wgpu, datashard_devices) 
          for wgpu in datashard_devices]

    caching_devices = None
      
  elif sync and ps_replicas > 0:
    # compute on ps
    datashard_devices = [
        _replica_device_setter(d) for d in ps_devices(all_workers=all_workers)
    ]
    if ps_gpu > 0 and ps_replicas > 1:
      caching_devices = [
          ps_job + "/task:%d/CPU:0" % d
          for (d, _) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      caching_devices = None
  else:
    # compute on worker - this is either a single-worker setup or asynchronous
    # with parameter servers.
    if worker_gpu > 1:
      datashard_devices = [
          _replica_device_setter(worker_job + "/GPU:%d" % d)
          for d in _gpu_order(worker_gpu)
      ]
      caching_devices = None
    else:
      datashard_devices = [_replica_device_setter(worker_job)]
      caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  tf.logging.info("ps_devices: %s", ps_devices(all_workers=all_workers))
  return DataParallelism(
      datashard_devices,
      caching_devices=caching_devices,
      daisy_chain_variables=daisy_chain_variables,
      ps_devices=ps_devices(all_workers=all_workers))


class DataParallelism(object):
  """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in range(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=True,
               caching_devices=None,
               daisy_chain_variables=False,
               ps_devices=None):
    """Create a Parallelism.

    Args:
      device_names_or_functions: A list of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.
      ps_devices: list<str>, list of devices for experts.

    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._maybe_repeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables
    self._ps_devices = ps_devices or [""]

  def __call__(self, fn, *args, **kwargs):
    """A parallel set of function calls (using the specified devices).

    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.

    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
    # Construct lists or args and kwargs for each function.
    if args:
      my_args = transpose_list_of_lists(
          [self._maybe_repeat(arg) for arg in args])
    else:
      my_args = [[] for _ in range(self.n)]
    my_kwargs = [{} for _ in range(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._maybe_repeat(v)
      for i in range(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._maybe_repeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    tensor_to_var = {}
    for i in range(self.n):

      def daisy_chain_getter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          last_device_v = cache[name]
          var = tensor_to_var[last_device_v]
          v = tf.identity(last_device_v)
        else:
          var = getter(name, *args, **kwargs)
          v = var.read_value()

        # keep track of the original variable
        tensor_to_var[v] = var
        _add_variable_proxy_methods(tensor_to_var[v], v)
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def caching_getter(getter, name, *args, **kwargs):
        """Cache variables on device."""
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]

        v = getter(name, *args, **kwargs)
        with tf.device(self._caching_devices[i]):
          ret = v.read_value()
        _add_variable_proxy_methods(v, ret)
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = daisy_chain_getter
      elif self._caching_devices[i]:
        custom_getter = caching_getter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope("parallel_%d" % i):
        with tf.variable_scope(
            tf.get_variable_scope() if self._reuse else "parallel_%d" % i,
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          # TODO(noam, epot, avaswani)
          # Allows for passing no device in case you want to default to the
          # existing device. This is needed when we put all experts on a single
          # device, for example in local_moe.
          if self._devices[i] != DEFAULT_DEV_STRING:
            with tf.device(self._devices[i]):
              outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
          else:
            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
    if isinstance(outputs[0], tuple):
      outputs = list(zip(*outputs))
      outputs = tuple([list(o) for o in outputs])
    return outputs

  @property
  def n(self):
    return self._n

  @property
  def devices(self):
    return self._devices

  @property
  def ps_devices(self):
    return self._ps_devices

  def _maybe_repeat(self, x):
    """Utility function for processing arguments that are singletons or lists.

    Args:
      x: either a list of self.n elements, or not a list.

    Returns:
      a list of self.n elements.
    """
    if isinstance(x, list):
      assert len(x) == self.n
      return x
    else:
      return [x] * self.n