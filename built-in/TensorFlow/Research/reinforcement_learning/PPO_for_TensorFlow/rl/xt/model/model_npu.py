# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
"""

from __future__ import division, print_function

import os

from xt.model.pb_format import pb_model
from xt.model.tf_compat import K, tf

os.environ["KERAS_BACKEND"] = "tensorflow"


class XTModel(object):
    """
    Model Base class for model module.
    Owing to the same name to Keras.Model, set `XTModel` as the base class.
    User could inherit the XTModel, to implement their model.
    """
    def __init__(self, model_info):
        """
        To avoid the compatibility problems about tensorflow's versions.
        Model class will hold their graph&session within itself.
        Now, we used the keras's API to create models.
        :param model_info:
        """
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        if model_info.get("type", "actor") is "learner":
            print(model_info)
            print("i am learner!!!!!!!!!!1")
            from npu_bridge.estimator import npu_ops
            from npu_bridge.estimator.npu.npu_config import NPURunConfig
            from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
            from npu_bridge.hccl import hccl_ops
            from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

            session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

            custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add(
            )
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["enable_data_pre_proc"].b = True
            custom_op.parameter_map["mix_compile_mode"].b = False
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["min_group_size"].b = 1
            # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

        self.graph = tf.Graph()
        with self.graph.as_default():
            sess = tf.Session(config=session_config)
            self.sess = sess
            K.set_session(self.sess)
            self.model_format = model_info.get('model_format')
            self.model = self.create_model(model_info)
            if 'init_weights' in model_info:
                model_name = model_info['init_weights']
                try:
                    self.load_model(model_name)
                    print("load weight: {} success.".format(model_name))
                except BaseException:
                    print("load weight: {} failed!".format(model_name))

    def create_model(self, model_info):
        """abstract method for creating model"""
        raise NotImplementedError

    def predict(self, state):
        """
        Do predict use the newest model.
        :param state:
        :return:
        """
        with self.graph.as_default():
            K.set_session(self.sess)
            return self.model.predict(state)

    def train(self, state, label):
        """train the model"""
        with self.graph.as_default():
            K.set_session(self.sess)
            loss = self.model.train_on_batch(state, label)
            return loss

    def get_weights(self):
        """return the weights of the model"""
        with self.graph.as_default():
            K.set_session(self.sess)
            return self.model.get_weights()

    def set_weights(self, weights):
        """set the new weights"""
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.set_weights(weights)

    def get_grad(self, data):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.get_grad(data)

    def save_model(self, file_name):
        """save weights into .h5 file"""
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.save_weights(file_name + ".h5", overwrite=True)
        if self.model_format == 'pb':
            pb_model(self.model, file_name)
        return file_name + ".h5"

    def load_model(self, model_name, by_name=False):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.load_weights(model_name, by_name)
