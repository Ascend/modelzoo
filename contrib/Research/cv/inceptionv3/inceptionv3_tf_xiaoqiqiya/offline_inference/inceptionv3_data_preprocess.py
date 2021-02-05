# -*- coding: UTF-8 -*-
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
import tensorflow as tf
import os
import numpy as np
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_training = False
batch_size = 32
epochs = 1
image_num = 50000


def preprocess_for_eval(image,
                        height,
                        width,
                        central_fraction=0.875,
                        scope=None,
                        central_crop=True,
                        use_grayscale=False):
    """Prepare one image for evaluation.
    If height and width are specified it would output an image with that size by
    applying resize_bilinear.
    If central_fraction is specified it would crop the central fraction of the
    input image.
    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
      central_crop: Enable central cropping of images during preprocessing for
        evaluation.
      use_grayscale: Whether to convert the image from RGB to grayscale.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if use_grayscale:
            image = tf.image.rgb_to_grayscale(image)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_crop and central_fraction:
            image = tf.image.central_crop(
                image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def _parse_read(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
               }
               
    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features["label"]
    images = tf.image.decode_jpeg(parsed_features["image"])
    h = tf.cast(parsed_features['height'], tf.int64)
    w = tf.cast(parsed_features['width'], tf.int64)
    c = tf.cast(parsed_features['channels'], tf.int64)
    images = tf.reshape(images, [h, w, 3])
    images = tf.cast(images, tf.float32)
    images = images/255.
    images1 = preprocess_for_eval(images,299,299,0.80)
    images2 = preprocess_for_eval(images,299,299,0.85)
    images3 = preprocess_for_eval(images,299,299, 0.9)
    images4 = preprocess_for_eval(images,299,299,0.95)
    images5 = preprocess_for_eval(images,299,299,0.925)
    return images1,images2,images3,images4,images5,label

if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    output_path += "/"
   #=====================================================#
    clear = True
    if clear:
        os.system("rm -rf "+output_path+"data")
        os.system("rm -rf "+output_path+"label")
    if os.path.isdir(output_path+"data"):
        pass
    else:
        os.makedirs(output_path+"data")
    if os.path.isdir(output_path+"label"):
        pass
    else:
        os.makedirs(output_path+"label")
   #=====================================================#
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_read,num_parallel_calls=4)
    dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    label = []
    count = 0
    for step in range(int(image_num / batch_size)):
        count = count + 1
        x_in, y_in = sess.run([images_batch, labels_batch])
        y_in = np.squeeze(y_in, 1)+1
        x_in.tofile(output_path+"data/"+str(step)+".bin")
        label += y_in.tolist()
    label = np.array(label)
    print("共 ",len(label)," 数据")
    np.save(output_path + "label/imageLabel.npy", label)
    print("[info]  data bin ok")
