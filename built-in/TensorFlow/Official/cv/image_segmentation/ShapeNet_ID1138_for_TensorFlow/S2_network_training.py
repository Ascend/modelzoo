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
import argparse
import h5py
import numpy as np
import tensorflow as tf

print(tf.version.VERSION)

#############npu modify start###############
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess= tf.compat.v1.Session(config=npu_config_proto(config_proto=config))
global_config = tf.ConfigProto(log_device_placement=False)
custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
npu_keras_sess = set_keras_session_npu_config(config=global_config)
#############npu modify end###############

import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

#%% load GPGL functions
import fun_network as fun
#%% global settings
NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
SIZE_IMG = SIZE_SUB*SIZE_SUB

####################### set params ###########################
parser = argparse.ArgumentParser(description="train params")
parser.add_argument("--train_steps", type=int, default=15000, help='train_steps.')
parser.add_argument("--train_batch_size", type=int, default=4, help='train_batch_size.')
parser.add_argument("--train_epochs", type=int, default=100, help='train_epochs.')
parser.add_argument("--dataset_path", type=str, default="ShapeNet_prepro.hdf5", help='dataset_path.')
parser.add_argument("--save_path", type=str, default="ShapeNet_model.h5", help='save_path.')
parser.add_argument("--val_batch_size", type=int, default=1, help='val_batch_size.')
parser.add_argument("--val_steps", type=int, default=500, help='val_steps.')
args = parser.parse_args()
####################### set params ###########################

#%%
f = h5py.File(args.dataset_path,'r')

####################### add param: train_step ###########################
#%% initialize the data loader
train_loader = fun.Dataloder(f,'train', SIZE_IMG, batch_size=args.train_batch_size, shuffle=1, step=args.train_steps)
val_loader = fun.Dataloder(f,'val', SIZE_IMG, batch_size=args.val_batch_size, shuffle=0, step=args.val_steps)
####################### add param: train_step ###########################

#%%
input_shape = [SIZE_IMG,SIZE_IMG,3]
inputs1 = L.Input(shape=input_shape)
x = inputs1

mask = L.Lambda(lambda x: fun.build_mask(x),name='mask')(inputs1)
mask = L.Lambda(lambda x: K.expand_dims(x),name='expend')(mask)
mask1 = L.Lambda(lambda x: K.tile(x, (1, 1, 1, 50)),name='tile')(mask)
x0 = fun.Inception(x,64,name='x0')(x)

x1 = L.MaxPool2D(pool_size=(SIZE_SUB,SIZE_SUB),strides=(SIZE_SUB,SIZE_SUB),padding='valid')(x0)

x1 = fun.Inception(x1,128,name='x1')(x1)
x2 = L.MaxPool2D(pool_size=(SIZE_TOP,SIZE_TOP),strides=(SIZE_TOP,SIZE_TOP),padding='valid')(x1)
xg = x2

xg = L.Dense(256,activation='relu',name='x2')(xg)

y2 = xg
y1 = L.UpSampling2D(size=(SIZE_TOP, SIZE_TOP))(y2)
y1 = L.Concatenate()([x1,y1])
y1 = fun.Inception(y1,128,name='y1')(y1)

y0 = L.UpSampling2D(size=(SIZE_SUB, SIZE_SUB))(y1)
y0 = L.Concatenate()([x0,y0])
y0 = fun.Inception(y0,64,name='y0')(y0)

y = L.Dense(50,activation='softmax')(y0)

ouputs = y
ouputs = L.Multiply()([ouputs,mask1])
not_mask = L.Lambda(lambda x: 1-x)(mask)

ouputs = L.Concatenate(name="segment_out")([ouputs,not_mask])
model = keras.Model(inputs=inputs1, outputs=[ouputs])
print(model.summary())
#%%
opti = npu_keras_optimizer(keras.optimizers.Adam(1e-4))
#model.compile(opti,loss ='categorical_crossentropy',metrics=fun.iou)
model.compile(opti,loss ='categorical_crossentropy')

#%%
print("-------------------------------")
print("*******************************")
print("*****",args.save_path,"*****")
print("*******************************")
print("-------------------------------")
checkpointer = keras.callbacks.ModelCheckpoint(filepath=args.save_path, monitor='val_iou',mode='max', verbose=1, save_best_only=True,save_weights_only=False)

# history = model.fit(x=train_loader, validation_data=val_loader, epochs = args.train_epochs, verbose=1, callbacks=[checkpointer])
history = model.fit(x=train_loader, epochs = args.train_epochs, verbose=1, callbacks=[checkpointer])

print('====save model====')
model.save_weights('./ckpt_gpu/model_weights.h5')
model.save('./ckpt_gpu/model.h5')

#print("Best pixel wise accuracy",max(history.history['iou']))

#np.savetxt("ShapeNet_training_statistics.csv", np.vstack((history.history['loss'],history.history['iou'],history.history['val_loss'],history.history['val_iou'])).T , delimiter=",")
np.savetxt("ShapeNet_training_statistics.csv", np.vstack(history.history['loss']).T , delimiter=",")

close_session(npu_keras_sess)

