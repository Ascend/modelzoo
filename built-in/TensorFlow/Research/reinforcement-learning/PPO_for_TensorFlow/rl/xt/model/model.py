# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
model base
"""
import os
import glob
from xt.model.tf_compat import tf, K
from xt.model.pb_format import pb_model

os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

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
        use_npu = True
        if use_npu:
            session_config = tf.ConfigProto(
                allow_soft_placement=True,
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

                custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
                custom_op.name = "NpuOptimizer"
                custom_op.parameter_map["enable_data_pre_proc"].b = True
                custom_op.parameter_map["mix_compile_mode"].b = False
                custom_op.parameter_map["use_off_line"].b = True
                custom_op.parameter_map["min_group_size"].b = 1
                #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        else:
            session_config = tf.ConfigProto()
        self.graph = tf.Graph()

        with self.graph.as_default():
            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth = True
            sess = tf.Session(config=session_config)
            self.sess = sess
            K.set_session(self.sess)
            self.model_format = model_info.get('model_format')
            self.max_to_keep = model_info.get("max_to_keep", 100)
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
        # check max model file to keep
        check_keep_model(os.path.dirname(file_name), self.max_to_keep)

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


def check_keep_model(model_path, keep_num):
    target_file = glob.glob(os.path.join(model_path, "actor*".format(model_path)))
    if len(target_file) > keep_num:
        to_rm_model = sorted(target_file, reverse=True)[keep_num:]
        for item in to_rm_model:
            os.remove(item)
