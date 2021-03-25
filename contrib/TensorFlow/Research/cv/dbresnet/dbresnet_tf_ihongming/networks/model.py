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
# Copyright 2020 Huawei Technologies Co., Ltd
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
import tensorflow as tf

from networks import resnet_v1
from tensorflow.contrib import slim


def dbnet(image_input, input_size=640, k=50, is_training=True, scope="resnet_v1_50"):
    
    with tf.name_scope("resnet_layer"):
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=1e-5)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=image_input, is_training=is_training, scope=scope)
        
        C2, C3, C4, C5 = end_points['pool2'], end_points['pool3'], end_points['pool4'], end_points['pool5']
       

    with tf.name_scope("detector_layer"):
        filter_in2 = tf.get_variable("filter_in2", [1, 1, 64, 256],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        in2 = tf.nn.conv2d(C2, filter=filter_in2, strides=[1, 1, 1, 1], padding='SAME', name='in2')

        filter_in3 = tf.get_variable("filter_in3", [1, 1, 256, 256],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        in3 = tf.nn.conv2d(C3, filter=filter_in3, strides=[1, 1, 1, 1], padding='SAME', name='in3')

        filter_in4 = tf.get_variable("filter_in4", [1, 1, 512, 256],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        in4 = tf.nn.conv2d(C4, filter=filter_in4, strides=[1, 1, 1, 1], padding='SAME', name='in4')

        filter_in5 = tf.get_variable("filter_in5", [1, 1, 2048, 256],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        in5 = tf.nn.conv2d(C5, filter=filter_in5, strides=[1, 1, 1, 1], padding='SAME', name='in5')

        out4 = tf.add(in4, tf.image.resize_nearest_neighbor(in5, size=[tf.shape(in5)[1] * 2, tf.shape(in5)[2] * 2]),
                      name='out4')

        out3 = tf.add(in3, tf.image.resize_nearest_neighbor(out4, size=[tf.shape(out4)[1] * 2, tf.shape(out4)[2] * 2]),
                      name='out3')

        out2 = tf.add(in2, tf.image.resize_nearest_neighbor(out3, size=[tf.shape(out3)[1] * 2, tf.shape(out3)[2] * 2]),
                      name='out2')

        filter_p5 = tf.get_variable("filter_p5", [3, 3, 256, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        in5_t = tf.nn.conv2d(in5, filter=filter_p5, strides=[1, 1, 1, 1], padding='SAME', name='in5_t')
        P5 = tf.image.resize_nearest_neighbor(in5_t, size=[tf.shape(in5_t)[1] * 8, tf.shape(in5_t)[2] * 8], name="P5")

        filter_p4 = tf.get_variable("filter_p4", [3, 3, 256, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        out4_t = tf.nn.conv2d(out4, filter=filter_p4, strides=[1, 1, 1, 1], padding='SAME', name='out4_t')
        P4 = tf.image.resize_nearest_neighbor(out4_t, size=[tf.shape(out4_t)[1] * 4, tf.shape(out4_t)[2] * 4], name="P4")

        filter_p3 = tf.get_variable("filter_p3", [3, 3, 256, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        out3_t = tf.nn.conv2d(out3, filter=filter_p3, strides=[1, 1, 1, 1], padding='SAME', name='out3_t')
        P3 = tf.image.resize_nearest_neighbor(out3_t, size=[tf.shape(out3_t)[1] * 2, tf.shape(out3_t)[2] * 2],
                                              name="P3")

        filter_p2 = tf.get_variable("filter_p2", [3, 3, 256, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        P2 = tf.nn.conv2d(out2, filter=filter_p2, strides=[1, 1, 1, 1], padding='SAME', name='P2')

        fuse = tf.concat([P5, P4, P3, P2], axis=3)

        # probability map
        filter_probability = tf.get_variable("filter_probability", [3, 3, 256, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        p = tf.nn.conv2d(fuse, filter=filter_probability, strides=[1, 1, 1, 1], padding='SAME')
        p = tf.layers.batch_normalization(p, training=is_training, momentum=0.9)
        p = tf.nn.relu(p)

        filter_tr = tf.get_variable("filter_tr", [2, 2, 64, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        p = tf.nn.conv2d_transpose(p, output_shape=[tf.shape(p)[0], tf.shape(p)[1] * 2, tf.shape(p)[2] * 2, 64],
                                   filter=filter_tr,
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   )
        p = tf.layers.batch_normalization(p, training=is_training, momentum=0.9)
        p = tf.nn.relu(p)
        filter_tr2 = tf.get_variable("filter_tr2", [2, 2, 1, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        p = tf.nn.conv2d_transpose(p, output_shape=[tf.shape(p)[0], tf.shape(p)[1] * 2, tf.shape(p)[2] * 2, 1],
                                   filter=filter_tr2,
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        p = tf.nn.sigmoid(p)

        # threshold map
        filter_threshold = tf.get_variable("filter_threshold", [3, 3, 256, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        t = tf.nn.conv2d(fuse, filter=filter_threshold, strides=[1, 1, 1, 1], padding='SAME')
        t = tf.layers.batch_normalization(t, training=is_training, momentum=0.9)
        t = tf.nn.relu(t)

        filter_th = tf.get_variable("filter_th", [2, 2, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        t = tf.nn.conv2d_transpose(t, output_shape=[tf.shape(t)[0], tf.shape(t)[1] * 2, tf.shape(t)[2] * 2, 64],
                                   filter=filter_th,
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        t = tf.layers.batch_normalization(t, training=is_training, momentum=0.9)
        t = tf.nn.relu(t)
        filter_th2 = tf.get_variable("filter_th2", [2, 2, 1, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        t = tf.nn.conv2d_transpose(t, output_shape=[tf.shape(t)[0], tf.shape(t)[1] * 2, tf.shape(t)[2] * 2, 1],
                                   filter=filter_th2,
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        t = tf.nn.sigmoid(t)

        # approximate binary map
        b_hat = tf.reciprocal(1 + tf.exp(-k * (p - t)), name='thresh_binary')

        return p, t, b_hat


if __name__ == '__main__':
    image_input = tf.placeholder("float", shape=[None, 640, 640, 3], name="image_input")
    p, t, b_hat = dbnet(image_input)
