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
import numpy as np
import time
import matplotlib.pyplot as plt
# from pylab import rcParams

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator,NPUEstimatorSpec
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu import npu_loss_scale_optimizer
from npu_bridge.estimator.npu import npu_loss_scale_manager

def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print ("[%2d] %s %s = %s" % (idx, op.name, shape, count))
    total_count += int(count)
  print ("[Total] variable size: %s" % "{:,}".format(total_count))


def save_images_with_nll(images, nlls):
  num_images = images.shape[0]
  num_images_per_row = 4
  num_images_per_column = (num_images + num_images_per_row - 1) // num_images_per_row
  idx = 0
  for i in range(num_images_per_column):
    for j in range(num_images_per_row):
      plt.subplot2grid((num_images_per_column,num_images_per_row),(i, j))
      plt.axis('off')
      plt.imshow(images[idx])
      plt.title('%f' % nlls[idx])      
      idx += 1
      if idx >= num_images:
        plt.savefig('test_results/samples_%s.png' % time.strftime("%m_%d_%H_%M_%S"), bbox_inches='tight')
        return
  

