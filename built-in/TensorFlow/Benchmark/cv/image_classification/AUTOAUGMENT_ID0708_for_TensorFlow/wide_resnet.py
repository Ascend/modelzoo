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
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Add, GlobalAveragePooling2D, Dropout
from tensorflow.keras import regularizers
from npu_bridge.estimator.npu import npu_convert_dropout
weight_decay = 5e-4


def conv3x3(input, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2D(out_planes, kernel_size=3, strides=stride,
                    padding='same', use_bias=False, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(input)


def conv1x1(input, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2D(out_planes, kernel_size=1, strides=stride,
                    padding='same', use_bias=False, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(input)


def BasicBlock(input, planes, dropout, stride=1):
    #inplanes = input._keras_shape[3]
    inplanes = input.shape[3]

    out = BatchNormalization()(input)
    out = Activation('relu')(out)
    out = conv3x3(out, planes, stride)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = conv3x3(out, planes)

    if stride != 1 or inplanes != planes:
        shortcut = conv1x1(input, planes, stride)
    else:
        shortcut = out

    out = Add()([out, shortcut])

    return out


def WideResNet(depth, width, num_classes=10, dropout=0.3):
    layer = (depth - 4) // 6

    input = Input(shape=(32, 32, 3))

    x = conv3x3(input, 16)
    for _ in range(layer):
        x = BasicBlock(x, 16*width, dropout)
    x = BasicBlock(x, 32*width, dropout, 2)
    for _ in range(layer-1):
        x = BasicBlock(x, 32*width, dropout)
    x = BasicBlock(x, 64*width, dropout, 2)
    for _ in range(layer-1):
        x = BasicBlock(x, 64*width, dropout)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(input, output)
    model.summary()

    return model

