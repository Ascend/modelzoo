# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_training = False
batch_size = 1
epochs = 1
image_num = 736



def _parse_read(tfrecord_file):
    features = {
        'image':
            tf.io.FixedLenFeature((), tf.string),
        "label":
            tf.io.FixedLenFeature((), tf.string),
        'height':
            tf.io.FixedLenFeature((), tf.int64),
        'width':
            tf.io.FixedLenFeature((), tf.int64),
        'channels':
            tf.io.FixedLenFeature((), tf.int64)
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(parsed['label'], tf.uint8)
    label = tf.reshape(label, [parsed['height'], parsed['width']])
    h_pad = 512 - parsed['height']
    w_pad = 512 - parsed['width']
    image_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0))
    label_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2))
    image = tf.pad(image, image_padding, mode='constant', constant_values=0)
    label = tf.pad(label, label_padding, mode='constant', constant_values=0)
    image = image - [122.67891434, 116.66876762, 104.00698793]
    image = image / 255.
    return image, label, parsed['height'], parsed['width']



if __name__ == '__main__':

    data_path = sys.argv[1]
    output_path = sys.argv[2]
    output_path += "/"


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

    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, hs, ws = iterator.get_next()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    image = []
    label = []
    hwlist = []
    for step in range(int(image_num / batch_size)):
        x_in, y_in,h,w = sess.run([images_batch, labels_batch, hs, ws])
        hwlist.append([h,w])
        label.append(y_in)
        x_in.tofile(output_path+"data/"+str(step)+".bin")
    label = np.array(label)
    hwlist = np.array(hwlist)
    np.save(output_path + "label/label.npy", label)
    np.save(output_path + "label/hwlist.npy", hwlist)
    print("[info]  data bin ok")