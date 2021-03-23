"""
Function: Load paired images for face anti-spoofing
Author: AJ
Date: 2020.12.25
"""
import tensorflow as tf
import numpy as np
import scipy, os, random, glob
from scipy import misc
import cv2
suffix = '.jpg'

def check_if_exist(path):
    """function: Determine if the file exists"""
    return os.path.exists(path)
def make_if_not_exist(path):
    """function: Determine if the file exists, and make"""
    if not os.path.exists(path):
        os.makedirs(path)
def align_imagee_py(image_decoded, target_image_size):
    image_decoded = image_decoded.decode(encoding='UTF-8')
    image_size = cv2.imread(image_decoded).shape
    size_h, size_w = image_size[0], image_size[1]
    resize_flag = False
    if (size_h < target_image_size[0]) or (size_w < target_image_size[1]):
        resize_flag = True
        size_h, size_w = target_image_size[0], target_image_size[1]
        size_h = np.array(size_h).astype('int64')
        size_w = np.array(size_w).astype('int64')
    return size_h, size_w, resize_flag
def resize_py_image(image_decoded, image_size):
    image_size = tuple(image_size)
    image_resized = cv2.resize(image_decoded, image_size)
    return image_resized
def get_aug_flag(flag, set):
    return tf.equal(set, flag)
def is_preprocess_imagenet(image):
    image = tf.cast(image, tf.float32)
    return image
def random_rotate_image(image, max_angle, domain):
    domain = domain.decode(encoding='UTF-8')
    if domain == 'oulu':
        modal = 'ccrop'
    else:modal = domain
    if ('color' == modal) or ('profile' == modal) or ('ccrop' == modal):
        angle = np.random.uniform(low=-max_angle, high=max_angle)
    else:angle = max_angle
    return scipy.ndimage.interpolation.rotate(image, angle), angle

######  group ######
def random_crop_py_image(im_input, color_size):
    H = im_input.shape[0]  # 300
    W = im_input.shape[1]  # 300
    if H >= color_size[0]:
        hh = np.random.random_integers(0, H - color_size[0])
        hh_path_size = color_size[0]
    else:
        hh = 0
        hh_path_size = H
    if W >= color_size[1]:
        ww = np.random.random_integers(0, W - color_size[0])
        ww_path_size = color_size[0]
    else:
        ww = 0
        ww_path_size = W
    input_patch = im_input[hh:hh + hh_path_size, ww:ww + ww_path_size, :]
    # print('1', im_input.shape, color_size[0], hh, hh + hh_path_size, input_patch.shape)
    return input_patch, hh, ww
def fix_crop_py_image(im_input, color_size, hh, ww):
    input_patch = im_input[hh:hh + color_size[0], ww:ww + color_size[1], :]
    # print('2', im_input.shape, color_size[0], hh, hh + color_size[0], input_patch.shape)
    return input_patch

def replace_py(image_name_1, domain):
    image_name_1 = image_name_1.decode(encoding='UTF-8')
    domain = domain.decode(encoding='UTF-8')
    if domain == 'oulu':
        modal_1, modal_2 = 'ccrop', 'prnet'
        image_name_2 = image_name_1.replace(os.path.basename(image_name_1), 'prn_depth' + suffix)
    image_name_2 = image_name_2.replace(modal_1, modal_2)
    if not check_if_exist(image_name_2):
        return image_name_1
    return image_name_2
def depth_image_label(image, label):
    label = tf.to_float(label, name='to_float')
    return tf.multiply(image, tf.subtract(1.0, label))

def distort_color(image, alpha=8, beta=0.2, gamma=0.05):
    image = tf.image.random_brightness(image, max_delta=alpha / 255)
    image = tf.image.random_contrast(image, lower=1.0 - beta, upper=1.0 + beta)
    image = tf.image.random_hue(image, max_delta=gamma)
    image = tf.image.random_saturation(image, lower=1.0 - beta, upper=1.0 + beta)
    return image

class Dataset():
    def __init__(self, config, images_list, labels_list, mode):
        self.config = config
        self.color_size = (config.color_image_size, config.color_image_size)
        self.depth_size = (config.depth_image_size, config.depth_image_size)
        self.buffer_size = int(len(images_list) / self.config.batch_size)
        self.seed = config.seed

        self.color_mean_div = []
        self.depth_mean_div = []
        disorder_para = []
        data_augment = []
        self.color_mean_div += (float(i) for i in config.color_mean)
        self.depth_mean_div += (float(i) for i in config.depth_mean)
        disorder_para += (float(i) for i in config.disorder_para)
        data_augment += (int(i) for i in config.data_augment)
        self.max_angle = data_augment[0]
        self.RANDOM_FLIP = data_augment[1]
        self.RANDOM_CROP = data_augment[2]
        self.RANDOM_COLOR = data_augment[3]
        self.c_alpha = disorder_para[0]
        self.c_beta = disorder_para[1]
        self.c_gamma = disorder_para[2]
        self.flag = 0
        self.input_tensors = self.inputs_for_training(images_list, labels_list)
        self.nextit = self.input_tensors.make_one_shot_iterator().get_next()

    def inputs_for_training(self, images_list, labels_list):
        dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list))
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=-1)
        dataset = dataset.shuffle(self.buffer_size).batch(self.config.batch_size).repeat(self.config.max_nrof_epochs)
        return dataset

    def _parse_function(self, filename, label):
        image_color_modal = tf.image.decode_image(tf.io.read_file(filename), channels=3)
        # ### @0: get info of rgb_size
        # rgb_size_h, rgb_size_w, resize_flag = \
        #     tuple(tf.py_func(align_imagee_py, [filename, self.color_size], [tf.int64, tf.int64, tf.bool]))
        # ### @1: resize image_color_modal
        # image_color_modal = tf.cond(resize_flag,
        #                 lambda: tf.py_func(resize_py_image, [image_color_modal, (rgb_size_w, rgb_size_h)], tf.uint8),
        #                 lambda: image_color_modal)
        ### @2: RANDOM_ROTATE
        image_color_modal, angle = \
            tuple(tf.py_func(random_rotate_image, [image_color_modal, self.max_angle, 'oulu'], [tf.uint8, tf.double]))
        ### @3: Distort_color
        image_color_modal = tf.cond(get_aug_flag(self.RANDOM_COLOR, 1),
                        lambda: distort_color(image_color_modal, self.c_alpha, self.c_beta, self.c_gamma),
                        lambda: tf.identity(image_color_modal))
        ### @5: Crop_Resize
        # image_color_modal = tf.cond(get_aug_flag(self.RANDOM_CROP, 1),
        #                 lambda: tf.random_crop(image_color_modal, self.color_size + (3,), seed=self.seed),
        #                 lambda: tf.py_func(resize_py_image, [image_color_modal, self.color_size], tf.uint8))
        image_color_modal, hh, ww = \
            tuple(tf.py_func(random_crop_py_image, [image_color_modal, self.color_size], [tf.uint8, tf.int64, tf.int64]))

        ### FIXED_STANDARDIZATION
        image_color_modal = (tf.cast(image_color_modal, tf.float32) - self.color_mean_div[0]) / self.color_mean_div[1]
        image_color_modal = is_preprocess_imagenet(image_color_modal)
        image_color_modal.set_shape(self.color_size + (3,))

    ####### depth ########
        ### @1: resize image_depth_modal
        image_depth_modal = tf.image.decode_image(
            tf.io.read_file(tf.py_func(replace_py, [filename, 'oulu'], tf.string)), channels=3)
        # image_depth_modal = tf.py_func(resize_py_image, [image_depth_modal, (rgb_size_w, rgb_size_h)], tf.uint8)

        ### @2: RANDOM_ROTATE
        image_depth_modal, _ = \
            tuple(tf.py_func(random_rotate_image, [image_depth_modal, angle, 'depth'], [tf.uint8, tf.double]))
        ### @3: Distort_color
        image_depth_modal = tf.cond(get_aug_flag(self.RANDOM_COLOR, 1),
                        lambda: distort_color(image_depth_modal, self.c_alpha, self.c_beta, self.c_gamma),
                        lambda: tf.identity(image_depth_modal))
        ### @5: Crop_Resize
        # image_depth_modal = tf.cond(get_aug_flag(self.RANDOM_CROP, 1),
        #                 lambda: tf.image.random_crop(image_depth_modal, self.color_size + (3,), seed=self.seed),
        #                 lambda: tf.py_func(resize_py_image, [image_depth_modal, self.color_size], tf.uint8))
        image_depth_modal = \
            tf.py_func(fix_crop_py_image, [image_depth_modal, self.color_size, hh, ww], tf.uint8)

        image_depth_modal = tf.py_func(resize_py_image, [image_depth_modal, self.depth_size], tf.uint8)
        ### FIXED_STANDARDIZATION
        image_depth_modal = (tf.cast(image_depth_modal, tf.float32) - self.depth_mean_div[0]) / self.depth_mean_div[1]
        image_depth_modal = is_preprocess_imagenet(image_depth_modal)
        image_depth_modal.set_shape(self.depth_size + (3,))
        ###
        depth_label = depth_image_label(image_depth_modal, label)
        depth_label.set_shape(self.depth_size + (3,))
        domain = 'oulu'
        return domain, filename, image_color_modal, image_depth_modal, depth_label, label

