import mindspore.dataset as de
import numpy as np
from .dependency.datasets import get_dataset
from .dependency.datasets import build_sampler
import mmcv
import time
import cv2 as cv
from .dependency.datasets.extra_aug import RandomCrop, PhotoMetricDistortion, Expand 

def rescale_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data, scale_factor = mmcv.imrescale(img, (1280, 768), return_scale=True)
    if img_data.shape[0] > 768:
        img_data, scale_factor2 = mmcv.imrescale(img_data, (768, 760), return_scale=True)
        scale_factor = scale_factor*scale_factor2
    img_shape = np.append(img_shape, scale_factor)
    img_shape = np.asarray(img_shape,dtype=np.float32)
    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return  (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def resize_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (1280, 768), return_scale=True)
    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (768, 1280, 1.0)
    img_shape = np.asarray(img_shape,dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return  (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def resize_column_test(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (1280, 768), return_scale=True)
    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape,dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return  (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def impad_to_multiple_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = mmcv.impad(img, (768, 1280))
    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def imnormalize_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = mmcv.imnormalize(img, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], True)
    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def flip_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = img
    img_data = mmcv.imflip(img_data)
    flipped = gt_bboxes.copy()
    h, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1

    return  (img_data, img_shape, flipped, gt_label, gt_num)

def transpose_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float16)
    img_shape = img_shape.astype(np.float16)
    gt_bboxes = gt_bboxes.astype(np.float16)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def random_crop_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    random_crop = RandomCrop(min_crop_size=0.7)
    img_data, gt_bboxes, gt_label, gt_num = random_crop(img, gt_bboxes, gt_label, gt_num)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def photo_crop_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    random_photo = PhotoMetricDistortion()
    img_data, gt_bboxes, gt_label = random_photo(img, gt_bboxes, gt_label)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def expand_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    expand = Expand()
    img, gt_bboxes, gt_label = expand(img, gt_bboxes, gt_label)

    return (img, img_shape, gt_bboxes, gt_label, gt_num)
