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

import logging
import os
import glob

import cv2
import numpy as np
import random


def log(logflag, message, level='info'):
    """logging to stdout and logfile if flag is true"""
    if logflag:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'critical':
            logging.critical(message)


def create_dirs(target_dirs):
    """create necessary directories to save output files"""
    for dir_path in target_dirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def normalize_images(*arrays):
    """normalize input image arrays"""
    return [arr / 127.5 - 1 for arr in arrays]


def de_normalize_image(image):
    """de-normalize input image array"""
    return (image + 1) * 127.5


def save_image(FLAGS, images, phase, global_iter, save_max_num=5):
    """save images in specified directory"""
    if phase == 'train' or phase == 'pre-train':
        save_dir = FLAGS.train_url
    elif phase == 'inference':
        save_dir = FLAGS.inference_result_dir
        save_max_num = len(images)
    else:
        print('specified phase is invalid')

    for i, img in enumerate(images):
        if i >= save_max_num:
            break

        cv2.imwrite(
            save_dir + '/{0}_HR_{1}_{2}.jpg'.format(phase, global_iter, i),
            de_normalize_image(img))


def crop(img, FLAGS):
    """crop patch from an image with specified size"""
    img_h, img_w, _ = img.shape

    rand_h = np.random.randint(img_h - FLAGS.crop_size)
    rand_w = np.random.randint(img_w - FLAGS.crop_size)

    return img[rand_h:rand_h + FLAGS.crop_size,
               rand_w:rand_w + FLAGS.crop_size, :]


def data_augmentation(LR_images, HR_images, aug_type='horizontal_flip'):
    """data augmentation. input arrays should be [N, H, W, C]"""

    if aug_type == 'horizontal_flip':
        return LR_images[:, :, ::-1, :], HR_images[:, :, ::-1, :]
    elif aug_type == 'rotation_90':
        return np.rot90(LR_images, k=1, axes=(1, 2)), np.rot90(HR_images,
                                                               k=1,
                                                               axes=(1, 2))


def load_npz_data(FLAGS):
    """load array data from data_path"""
    return np.load(os.path.join(FLAGS.native_data, FLAGS.HR_npz_filename))['images'], \
           np.load(os.path.join(FLAGS.native_data, FLAGS.LR_npz_filename))['images']


def load_and_save_data(FLAGS, logflag):
    """make HR and LR data. And save them as npz files"""
    assert os.path.isdir(
        FLAGS.data_dir
    ) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(FLAGS.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_HR_image = []
    ret_LR_image = []

    for file in all_file_path:
        img = cv2.imread(file)
        filename = file.rsplit('/', 1)[-1]

        # crop patches if flag is true. Otherwise just resize HR and LR images
        if FLAGS.crop:
            for _ in range(FLAGS.num_crop_per_image):
                img_h, img_w, _ = img.shape

                if (img_h < FLAGS.crop_size) or (img_w < FLAGS.crop_size):
                    print(
                        'Skip crop target image because of insufficient size')
                    continue

                HR_image = crop(img, FLAGS)
                LR_crop_size = np.int(
                    np.floor(FLAGS.crop_size / FLAGS.scale_SR))
                LR_image = cv2.resize(HR_image, (LR_crop_size, LR_crop_size),
                                      interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(FLAGS.HR_data_dir + '/' + filename, HR_image)
                cv2.imwrite(FLAGS.LR_data_dir + '/' + filename, LR_image)

                ret_HR_image.append(HR_image)
                ret_LR_image.append(LR_image)
        else:
            HR_image = cv2.resize(img,
                                  (FLAGS.HR_image_size, FLAGS.HR_image_size),
                                  interpolation=cv2.INTER_LANCZOS4)
            LR_image = cv2.resize(img,
                                  (FLAGS.LR_image_size, FLAGS.LR_image_size),
                                  interpolation=cv2.INTER_LANCZOS4)

            cv2.imwrite(FLAGS.HR_data_dir + '/' + filename, HR_image)
            cv2.imwrite(FLAGS.LR_data_dir + '/' + filename, LR_image)

            ret_HR_image.append(HR_image)
            ret_LR_image.append(LR_image)

    assert len(ret_HR_image) > 0 and len(
        ret_LR_image) > 0, 'No availale image is found in the directory'
    log(logflag,
        'Data process : {} images are processed'.format(len(ret_HR_image)),
        'info')

    ret_HR_image = np.array(ret_HR_image)
    ret_LR_image = np.array(ret_LR_image)

    if FLAGS.data_augmentation:
        LR_flip, HR_flip = data_augmentation(ret_LR_image,
                                             ret_HR_image,
                                             aug_type='horizontal_flip')
        LR_rot, HR_rot = data_augmentation(ret_LR_image,
                                           ret_HR_image,
                                           aug_type='rotation_90')

        ret_LR_image = np.append(ret_LR_image, LR_flip, axis=0)
        ret_HR_image = np.append(ret_HR_image, HR_flip, axis=0)
        ret_LR_image = np.append(ret_LR_image, LR_rot, axis=0)
        ret_HR_image = np.append(ret_HR_image, HR_rot, axis=0)

        del LR_flip, HR_flip, LR_rot, HR_rot

    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.HR_npz_filename,
             images=ret_HR_image)
    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.LR_npz_filename,
             images=ret_LR_image)

    return ret_HR_image, ret_LR_image


def load_one_iter(LR_files, HR_files):
    lr_data = []


def load_inference_data(FLAGS):
    """load data from directory for inference"""
    assert os.path.isdir(
        FLAGS.data_dir
    ) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(FLAGS.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_LR_image = []
    ret_filename = []

    for file in all_file_path:
        img = cv2.imread(file)
        img = normalize_images(img)
        ret_LR_image.append(img[0][np.newaxis, ...])

        ret_filename.append(file.rsplit('/', 1)[-1])

    assert len(
        ret_LR_image) > 0, 'No available image is found in the directory'

    return ret_LR_image, ret_filename


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        return imgs