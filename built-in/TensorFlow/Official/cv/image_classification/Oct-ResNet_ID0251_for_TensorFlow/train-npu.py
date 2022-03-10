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
import numpy as np
import os
import time

from tensorflow import keras
from scipy import linalg
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, apply_affine_transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from npu_bridge.estimator import npu_ops
from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.data.experimental.ops import threadpool
from model.resnet import resnet_v1, resnet_v2

class LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, batch_size, 
                 iterations_per_loop, 
                 display_every,
                 num_records):
        self.iter_times = []
        self.iterations_per_loop = iterations_per_loop
        self.batch_size = batch_size
        self.num_records = num_records
        self.display_every = display_every
        self.elapsed_secs = 0.
        self.count = 0
        
    def before_run(self, run_context):
        self.start_time = time.time()
        return tf.train.SessionRunArgs(fetches=[tf.train.get_global_step(), 'loss/add:0'])
        
    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.start_time
        self.iter_times.append(batch_time)
        self.elapsed_secs += batch_time
        self.count += 1
        global_step, loss = run_values.results
        if global_step == 1 or global_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.batch_size * self.iterations_per_loop / dt
            epoch = global_step * self.batch_size / self.num_records
            print('step:%6i  epoch:%5.1f  FPS:%7.1f  loss:%6.3f'  %
                             (global_step, epoch, img_per_sec, loss))
                             
            self.elapsed_secs = 0.
            self.count = 0
            
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

if __name__ == "__main__":
    import sys
    # Training parameters
    batch_size = 32  # orig paper trained all networks with batch_size=128
    epochs = 3
    lre = 1e-3
    num_classes = 10

    model_path = './model_dir'

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # num fo res blocks in each stack
    n = int(sys.argv[2])

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = int(sys.argv[1])

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Load the CIFAR10 data.
    if os.path.exists("/root/.keras/datasets/") == False:
        os.system("mkdir -p /root/.keras/datasets/")
    if os.path.exists("/root/.keras/datasets/cifar-10-batches-py.tar.gz") == False:
        os.system("cp ./cifar-10-batches-py.tar.gz /root/.keras/datasets/cifar-10-batches-py.tar.gz")
    if os.path.exists("/root/.keras/datasets/cifar-10-python.tar.gz") == False:
        os.system("cp ./cifar-100-python.tar.gz    /root/.keras/datasets/cifar-100-python.tar.gz")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lre), 
                  metrics=['accuracy'])

    def process_function(x_data,
                     image_size,
                     num_channels=3,
                     shear_range=0.,
                     zoom_range=0.,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     horizontal_flip=False,
                     data_format='channels_last'):
        
        img = x_data

        def data_augment(x):
            if data_format == 'channels_first':
                channel_axis = 0
                row_axis = 1
                col_axis = 2
            if data_format == 'channels_last':
                channel_axis = 2
                row_axis = 0
                col_axis = 1
            
            if isinstance(zoom_range, (float, int)):
                zr = [1 - zoom_range, 1 + zoom_range]
            elif (len(zoom_range) == 2 and
                all(isinstance(val, (float, int)) for val in zoom_range)):
                zr = [zoom_range[0], zoom_range[1]]
            else:
                zr = [1, 1]

            if rotation_range:
                theta = np.random.uniform(low=-rotation_range, high=rotation_range)
            else:
                theta = 0
            
            h, w = x.shape[row_axis], x.shape[col_axis]
            if height_shift_range:
                tx = np.random.uniform(low=-height_shift_range, high=height_shift_range) * h
            else:
                tx = 0
            if width_shift_range:
                ty = np.random.uniform(low=-width_shift_range, high=width_shift_range) * w
            else:
                ty = 0

            if shear_range:
                shear = np.random.uniform(low=-shear_range, high=shear_range)
            else:
                shear = 0

            if zr[0] == 1 and zr[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = np.random.uniform(low=zr[0], high=zr[1], size=2)

            x = apply_affine_transform(x, theta=theta, tx=tx, ty=ty,
                                    shear=shear, zx=zx, zy=zy,
                                    row_axis=row_axis,
                                    col_axis=col_axis,
                                    channel_axis=channel_axis)

            flip_horizontal = (np.random.random() < 0.5) * horizontal_flip
            if flip_horizontal:
                x = np.asarray(x).swapaxes(col_axis, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, col_axis)

            return x
        
        result_tensors = tf.py_func(data_augment, inp=[img], Tout=tf.float32)
        result_tensors.set_shape((image_size[0], image_size[1], num_channels))
        return result_tensors

    def input_fn(batch_size, image_size, x_data, labels, num_classes, is_train):
        if is_train:
            kwargs = {'image_size' : image_size,
                    'num_channels': 3,
                    'shear_range': 0.,
                    'zoom_range': 0.,
                    'rotation_range': 0.,
                    'width_shift_range': 0.,
                    'height_shift_range': 0.,
                    'horizontal_flip': False,
                    'data_format': 'channels_last'}
        else:
            kwargs = {'image_size' : image_size,
                    'num_channels': 3}
        
        img_ds = tf.data.Dataset.from_tensor_slices(x_data)
        img_ds = img_ds.map(lambda path: process_function(path, **kwargs), num_parallel_calls=256)

        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(labels, num_classes)
        label_ds = tf.data.Dataset.from_tensor_slices(y_train)
        ds = tf.data.Dataset.zip((img_ds, label_ds))
        ds = ds.shuffle(buffer_size=batch_size * 1, seed=np.random.randint(1e6)).batch(32, drop_remainder=True).repeat(100)
        ds = threadpool.override_threadpool(ds, threadpool.PrivateThreadPool(128, display_name='input_pipeline_thread_pool'))
        return ds

    run_config = NPURunConfig(enable_data_pre_proc=True,
                          save_checkpoints_steps=5000,
                          model_dir=model_path,
                          precision_mode='allow_mix_precision',
                          iterations_per_loop=10)

    est_resnet = keras_to_npu.model_to_npu_estimator(model, model_dir=model_path, config=run_config)
    K.clear_session()

    training_hooks = [LogSessionRunHook(32, 10, 1, 50000)]
    est_resnet.train(input_fn=lambda: input_fn(batch_size, input_shape, x_train, y_train, num_classes, True),
                     max_steps=4689,
                     hooks=training_hooks)

    est_resnet.evaluate(input_fn=lambda: input_fn(batch_size, input_shape, x_test, y_test, num_classes, False),
                        steps=313)