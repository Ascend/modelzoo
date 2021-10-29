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
"""Test resnet3d."""
from npu_bridge.npu_init import *
import pytest
from keras import backend as K
from resnet3d import Resnet3DBuilder


@pytest.fixture
def resnet3d_test():
    """resnet3d test helper."""
    def f(model):
        K.set_image_data_format('channels_last')
        model.compile(loss="categorical_crossentropy", optimizer=npu_keras_optimizer(tf.keras.optimizers.SGD()))
        assert True, "Failed to build with {}".format(K.image_data_format())
    return f


def test_resnet3d_18(resnet3d_test):
    """Test 18."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_18((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_18((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_resnet3d_34(resnet3d_test):
    """Test 34."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_34((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_34((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_resnet3d_50(resnet3d_test):
    """Test 50."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_50((224, 224, 224, 1), 1, 1e-2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_50((1, 512, 512, 256), 1, 1e-2)
    resnet3d_test(model)


def test_resnet3d_101(resnet3d_test):
    """Test 101."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_101((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_101((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_resnet3d_152(resnet3d_test):
    """Test 152."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_152((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_152((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_bad_shape():
    """Input shape need to be 4."""
    K.set_image_data_format('channels_last')
    with pytest.raises(ValueError):
        Resnet3DBuilder.build_resnet_152((224, 224, 224), 2)


def test_get_block():
    """Test get residual block."""
    K.set_image_data_format('channels_last')
    Resnet3DBuilder.build((224, 224, 224, 1), 2, 'bottleneck',
                          [2, 2, 2, 2], reg_factor=1e-4)
    assert True
    with pytest.raises(ValueError):
        Resnet3DBuilder.build((224, 224, 224, 1), 2, 'nullblock',
                              [2, 2, 2, 2], reg_factor=1e-4)

