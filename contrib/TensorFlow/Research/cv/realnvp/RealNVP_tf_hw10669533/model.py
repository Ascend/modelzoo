# Copyright 2019 Huawei Technologies Co., Ltd
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
import nn as nn

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator,NPUEstimatorSpec
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu import npu_loss_scale_optimizer
from npu_bridge.estimator.npu import npu_loss_scale_manager

layers = []

# Currently the model is constructed as described in the paper
# for the CIFAR10 model.
def construct_model_spec():
  global layers
  num_scales = 2
  for scale in range(num_scales-1):    
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
    layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale))
    layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_3' % scale))
    layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

  # final layer
  scale = num_scales-1
  layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
  layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
  layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
  layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_4' % scale))
  layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))


# the final dimension of the latent space is recorded here
# so that it can be used for constructing the inverse model
final_latent_dimension = []
def model_spec(x):
  counters = {}
  xs = nn.int_shape(x)
  sum_log_det_jacobians = tf.zeros(xs[0])    

  # corrupt data (Tapani Raiko's dequantization)
  y = x*0.5 + 0.5
  y = y*255.0
  corruption_level = 1.0
  y = y + corruption_level * tf.random_uniform(xs)
  y = y/(255.0 + corruption_level)

  #model logit instead of the x itself    
  alpha = 1e-5
  y = y*(1-alpha) + alpha*0.5
  jac = tf.reduce_sum(-tf.log(y) - tf.log(1-y), [1,2,3])
  y = tf.log(y) - tf.log(1-y)
  sum_log_det_jacobians += jac

  if len(layers) == 0:
    construct_model_spec()

  # construct forward pass    
  z = None
  jac = sum_log_det_jacobians
  for layer in layers:
    y,jac,z = layer.forward_and_jacobian(y, jac, z)

  z = tf.concat( [z,y],3,name='par1')

  # record dimension of the final variable
  global final_latent_dimension
  final_latent_dimension = nn.int_shape(z)
  z=tf.add(z,0,name="par2")
  jac=tf.add(jac,0,name='jac2')
  # z = tf.cast(z, z.dtype, name="par2")
  # jac = tf.cast(jac, jac.dtype, name="jac")
  return z,jac

def inv_model_spec(y):
  # construct inverse pass for sampling
  shape = final_latent_dimension
  z = tf.reshape(y, [-1, shape[1], shape[2], shape[3]])
  y = None
  for layer in reversed(layers):
    y,z = layer.backward(y,z)
    
  # inverse logit
  # x = 1.0/(1 + tf.exp(-y))
  x = tf.reciprocal(1 + tf.exp(-y))

  return x
    
  
