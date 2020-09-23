import tensorflow as tf
from tensorflow.contrib.image.python.ops import distort_image_ops
import math
import random

def decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')


def random_horizontal_flip(image, prob):
    if prob > random.random():
        image = tf.image.flip_left_right(image)
    return image


def decode_crop_and_resize(record, bbox, size, scale, ratio):
    with tf.name_scope('decode_crop_and_resize'):
        height = 224
        width = 224
        crop_ratio = 0.8
        initial_shape = [int(round(height / crop_ratio)),
                int(round(width / crop_ratio)), 3]
        jpeg_shape = tf.image.extract_jpeg_shape( record )

        bbox_begin, bbox_size, bbox = \
            tf.image.sample_distorted_bounding_box(
                ##initial_shape,
                tf.image.extract_jpeg_shape(record),
                #tf.shape(image),
                #bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
                bounding_boxes=bbox,
                # tf.zeros(shape=[1,0,4]), # No bounding boxes
                min_object_covered=0.1,
                aspect_ratio_range=ratio,
                area_range=scale,
                max_attempts=10,
                 #   seed=11 ,  # Need to set for deterministic results
                use_image_if_no_bounding_boxes=True)

         # Reassemble the bounding box in the format the crop op requires.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

        image = tf.image.decode_and_crop_jpeg( record, crop_window, channels=3 )
        image = tf.image.resize_images( image, [height, width] )

        return image


def parse_and_preprocess_image_record_me(record, bbox, training):
    #imgdata, label, bbox, text = deserialize_image_record(record)
    #label -= 1  # Change to 0-based (don't use background class)
    with tf.name_scope('preprocess'):
        if training:
            #image = tf.image.decode_jpeg(record, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
            #image = random_resized_crop(image, 224, (0.08, 1.0), (0.75, 1.333))
            image = decode_crop_and_resize(record, bbox, 224, (0.08, 1.0), (0.75, 1.333))
            image = random_horizontal_flip(image, 0.5)
            image = normalize(image)
        else:
            image = decode_jpeg(record, channels=3)
            image = tf.image.resize_images(image, [256, 256])
            image = tf.image.central_crop(image, 224.0/256)
            image = normalize(image)

    return image

def normalize(inputs):
     # imagenet_mean = [121.0, 115.0, 100.0]             #np.array([121, 115, 100], dtype=np.float32)
     # imagenet_std =  [70.0, 68.0, 71.0]                #np.array([70, 68, 71], dtype=np.float32)
     imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
     imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
     imagenet_mean = tf.expand_dims(tf.expand_dims(imagenet_mean, 0), 0)
     imagenet_std = tf.expand_dims(tf.expand_dims(imagenet_std, 0), 0)
     inputs = inputs - imagenet_mean          #tf.subtract(inputs, imagenet_mean)
     inputs = inputs * (1.0 / imagenet_std)
     #inputs = tf.multiply(inputs, 1. / imagenet_std)

     return inputs

