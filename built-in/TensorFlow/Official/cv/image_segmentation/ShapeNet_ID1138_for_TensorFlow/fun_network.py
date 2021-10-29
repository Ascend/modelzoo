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
import h5py
import numpy as np

import tensorflow as tf
print(tf.version.VERSION)

import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from scipy.ndimage import distance_transform_cdt as cdt

#%%
@tf.function
def iou(y_true,y_pred):
    y_true_digit = K.flatten(K.argmax(y_true,axis=-1))
    y_pred_digit = K.flatten(K.argmax(y_pred,axis=-1))
    classes_true = tf.unique(y_true_digit)[0]
    classes_pred = tf.unique(y_pred_digit)[0]
    classes = K.concatenate([classes_true,classes_pred])
    classes = tf.unique(classes)[0]
    classes = classes[classes!=50]

    iou = tf.constant(0,dtype=tf.float32)
    for i_class in classes:
        value = K.zeros_like(y_true_digit)+K.cast(i_class,tf.int64)
        mask_true = K.cast(K.equal(y_true_digit,value),K.floatx())
        mask_pred = K.cast(K.equal(y_pred_digit,value),K.floatx())

        tp = K.sum(K.minimum(mask_true,mask_pred))
        fp = K.sum(K.minimum(1-mask_true,mask_pred))
        fn = K.sum(K.minimum(mask_true,1-mask_pred))

        iou_class = (tp+1e-12)/(tp+fp+fn+1e-12)
        iou += iou_class
    num_classes = K.cast(len(classes),tf.float32)
    iou = iou/num_classes
    return iou

def build_mask(inputs):
    mask_mat = K.stop_gradient(K.abs(inputs))
    mask = K.stop_gradient(K.sum(mask_mat,axis=-1,keepdims=False))
    mask = K.stop_gradient(K.sign(mask))
    return mask

def Inception(inputs,filters,name):
    input_shape = inputs.shape[1:].as_list()
    input_tensor = L.Input(shape=input_shape)
    x0 = L.Conv2D(filters,(1,1),padding='same',activation='relu')(input_tensor)
    x1 = L.Conv2D(filters,(3,3),padding='same',activation='relu')(input_tensor)
    # x1 = L.Conv2D(filters,(3,3),padding='same',activation='relu')(x1)
    # x1 = L.Conv2D(filters,(3,3),padding='same',activation='relu')(x1)
    x = L.Concatenate()([x0,x1])
    model = keras.Model(inputs=input_tensor, outputs=x,name=name)
    return model

#%% define the data loader
class Dataloder(keras.utils.Sequence):
    ####################### add param: train_step ###########################
    def __init__(self, dataset, phase, SIZE_IMG, batch_size=1, shuffle=0, step=15000):
        self.dataset = dataset
        self.phase  = phase
        self.batch_size = batch_size
        self.step = min(step, 15000) * batch_size
        self.shuffle = shuffle
        self.x_set_all = self.dataset['x_'+self.phase]
        self.s_set_all = self.dataset['s_'+self.phase]
        self.p_set_all = self.dataset['p_'+self.phase]
        self.x_set = self.x_set_all[:self.step]
        self.s_set = self.s_set_all[:self.step]
        self.p_set = self.p_set_all[:self.step]
        self.SIZE_IMG  = SIZE_IMG
    ####################### add param: train_step ###########################
        self.num_data = len(self.x_set)
        self.indexes_data = np.arange(self.num_data)
        self.on_epoch_end()
        
        self.input_d = np.zeros([self.batch_size,self.SIZE_IMG,self.SIZE_IMG,3],dtype=np.float32)
        self.output_l = np.zeros([self.batch_size,self.SIZE_IMG,self.SIZE_IMG,51],dtype=np.float32)

    def __getitem__(self, i):
        idx_batch_start = i*self.batch_size
        for i_data in range(self.batch_size):
            idx_sample = self.indexes_data[idx_batch_start+i_data]

            x_pt = self.x_set[idx_sample]
            s_pt = self.s_set[idx_sample]
            p_pt = self.p_set[idx_sample]

            d_im = np.zeros([self.SIZE_IMG,self.SIZE_IMG,3],dtype=np.float32)
            l_im = np.zeros([self.SIZE_IMG,self.SIZE_IMG,1],dtype=np.float32)+50.

            d_im[p_pt[:,0],p_pt[:,1]] = x_pt
            l_im[p_pt[:,0],p_pt[:,1]] = s_pt

            self.input_d[i_data] = d_im
            self.output_l[i_data] = keras.utils.to_categorical(l_im,51)

        return self.input_d, self.output_l

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_data//self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes_data)

