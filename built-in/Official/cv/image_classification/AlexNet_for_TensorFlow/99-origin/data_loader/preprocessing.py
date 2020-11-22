import tensorflow as tf
from tensorflow.contrib.image.python.ops import distort_image_ops
import math
import random


def deserialize_image_record(record): 
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        bbox = tf.stack([obj['image/object/bbox/%s' % x].values
                         for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
        text = obj['image/class/text']
        return imgdata, label, bbox, text

def decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')


def crop_and_resize_image(config, image, height, width, 
                          distort=False, nsummary=10):
    with tf.name_scope('crop_and_resize'):
        # Evaluation is done on a center-crop of this ratio
        eval_crop_ratio = 0.8
        if distort:
            initial_shape = [int(round(height / eval_crop_ratio)),
                             int(round(width / eval_crop_ratio)),
                             3]
            jpeg_shape = tf.image.extract_jpeg_shape( image )

            bbox_begin, bbox_size, bbox = \
                tf.image.sample_distorted_bounding_box(
                    initial_shape,
                    bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
                    min_object_covered=config['min_object_covered'],
                    aspect_ratio_range=config['aspect_ratio_range'],
                    area_range=config['area_range'],
                    max_attempts=config['max_attempts'],
                    use_image_if_no_bounding_boxes=True)
            bbox = bbox[0, 0]  # Remove batch, box_idx dims

            y_min = tf.cast( bbox[0] * (tf.cast( jpeg_shape[0], tf.float32) ), tf.int32)
            x_min = tf.cast( bbox[1] * (tf.cast(jpeg_shape[1], tf.float32) ), tf.int32) 
            y_max = tf.cast( bbox[2] * (tf.cast(jpeg_shape[0], tf.float32) ), tf.int32)
            x_max = tf.cast( bbox[3] * (tf.cast(jpeg_shape[1], tf.float32) ), tf.int32)

            crop_height = y_max - y_min
            crop_width = x_max - x_min
            crop_window = tf.stack( [y_min, x_min, crop_height, crop_width] )
            image = tf.image.decode_and_crop_jpeg( image, crop_window, channels=3 )
            image = tf.image.resize_images( image, [height, width] )

        else:
            # Central crop
            image = decode_jpeg(image, channels=3)
            ratio_y = ratio_x = eval_crop_ratio
            bbox = tf.constant([0.5 * (1 - ratio_y), 0.5 * (1 - ratio_x),
                                0.5 * (1 + ratio_y), 0.5 * (1 + ratio_x)])
            image = tf.image.crop_and_resize(
               image[None, :, :, :], bbox[None, :], [0], [height, width])[0]
        
        return image


def parse_and_preprocess_image_record(config, record, height, width,
                                      brightness, contrast, saturation, hue,
                                      distort, nsummary=10, increased_aug=False, random_search_aug=False):
    with tf.name_scope('preprocess_train'):
            image = crop_and_resize_image(config, record, height, width, distort)
            if distort:
                image = tf.image.random_flip_left_right(image)
                if increased_aug:
                    image = tf.image.random_brightness(image, max_delta=brightness)  
                    image = distort_image_ops.random_hsv_in_yiq(image, 
                                                                lower_saturation=saturation, 
                                                                upper_saturation=2.0 - saturation, 
                                                                max_delta_hue=hue * math.pi)
                    image = tf.image.random_contrast(image, lower=contrast, upper=2.0 - contrast)
                    tf.summary.image('distorted_color_image', tf.expand_dims(image, 0))

            image = tf.clip_by_value(image, 0., 255.)
    image = normalize(image)
    image = tf.cast(image, tf.float16)
    return image


def random_horizontal_flip(image, prob):
    if prob > random.random():
        image = tf.image.flip_left_right(image)
    return image


def random_resized_crop(record, size, scale, ratio):
    with tf.name_scope('crop_and_resize'):
        height = 224
        width = 224
        eval_crop_ratio = 0.8
        initial_shape = [int(round(height / eval_crop_ratio)),
                int(round(width / eval_crop_ratio)), 3]
        jpeg_shape = tf.image.extract_jpeg_shape( record )

        bbox_begin, bbox_size, bbox = \
            tf.image.sample_distorted_bounding_box(
                initial_shape,
                bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
                aspect_ratio_range=ratio,
                area_range=scale,
                max_attempts=10,
                use_image_if_no_bounding_boxes=True)
        bbox = bbox[0, 0]  # Remove batch, box_idx dims

        y_min = tf.cast( bbox[0] * (tf.cast(jpeg_shape[0], tf.float32) ), tf.int32)
        x_min = tf.cast( bbox[1] * (tf.cast(jpeg_shape[1], tf.float32) ), tf.int32) 
        y_max = tf.cast( bbox[2] * (tf.cast(jpeg_shape[0], tf.float32) ), tf.int32)
        x_max = tf.cast( bbox[3] * (tf.cast(jpeg_shape[1], tf.float32) ), tf.int32)

        crop_height = y_max - y_min
        crop_width = x_max - x_min
        crop_window = tf.stack( [y_min, x_min, crop_height, crop_width] )
        image = tf.image.decode_and_crop_jpeg( record, crop_window, channels=3 )
        image = tf.image.resize_images( image, [height, width] )

        return image

def parse_and_preprocess_image_record_hxb(record, distort):

    with tf.name_scope('preprocess_train'):
        if distort:
            image = random_resized_crop(record, 224, (0.08, 1.0), (0.75, 1.333))
            image = random_horizontal_flip(image, 0.5)
            image = normalize(image)
        else:
            image = decode_jpeg(record, channels=3)
            image = tf.image.resize_images(image, [256, 256])
            image = tf.image.central_crop(image, 224.0/256)
            image = normalize(image)

    return image


def normalize(inputs):

     imagenet_mean = [123.675, 116.28, 103.53]             #np.array([121, 115, 100], dtype=np.float32)
     imagenet_std =  [58.395, 57.12, 57.375]                #np.array([70, 68, 71], dtype=np.float32)
     imagenet_mean = tf.expand_dims(tf.expand_dims(imagenet_mean, 0), 0)
     imagenet_std = tf.expand_dims(tf.expand_dims(imagenet_std, 0), 0)
     inputs = inputs - imagenet_mean
     inputs = inputs * (1.0 / imagenet_std)

     return inputs
