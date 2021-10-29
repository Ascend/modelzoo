#
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
#

from npu_bridge.npu_init import *
import tensorflow as tf
import os


def predict_proba(graph, ops, X, run):
    """Predict probabilities with a fitted model.

    Arguments:
        graph: A Tensorflow graph.
        ops: A dict of ops of the graph.
        X: A numpy array of shape [n_samples, image_size, image_size, 3]
            and of type 'float32', a batch of images with
            pixel values in range [0, 1].
        run: An integer that determines a folder where a fitted model
            is saved.

    Returns:
        predictions: A numpy array of shape [n_samples, n_classes]
            and of type 'float32'.
    """
    sess = tf.Session(config=npu_config_proto(), graph=graph)
    ops['saver'].restore(sess, os.path.join('saved', 'run' + str(run) + '/model'))

    feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
    predictions = sess.run(ops['predictions'], feed_dict)

    sess.close()
    return predictions
