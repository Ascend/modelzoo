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

import argparse
import glob
import os
import numpy as np
import cv2
from skimage.measure import compare_psnr, compare_ssim

'''
   功能：预测图与原高清图对比计算精度
   python3 evaluation.py --HR_data_dir ./DIV2K_test_100/DIV2K_train_HR_801_900 --inference_result ./DIV2K_test_predicted/DIV2K_train_LR_bicubic_801_900_X2/
'''


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def calculate_psnr(img1, img2, crop_border, test_y_channel=True):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border, test_y_channel=True):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    ssims = []
    if (img1.ndim == 3):
        for i in range(img1.shape[2]):
            ssims.append(_ssim(img1[..., i], img2[..., i]))
    else:
        ssims.append(_ssim(img1, img2))
    return np.array(ssims).mean()


def calc_measures(args, calc_psnr=True, calc_ssim=True):
    """calculate PSNR and SSIM for all HR images and their mean.
    These paired images should have the same filename.
    """

    HR_files = glob.glob(args.HR_data_dir + '/*')
    mean_psnr = []
    mean_ssim = []
    test_y_channel = True

    for file in HR_files:
        hr_img = cv2.imread(file).astype(np.float32) / 255
        filename = file.rsplit('/', 1)[-1]
        print("---filename----",filename)
        #filename = filename.split('.')[0]
        # file_split = filename.split('_')
        # filename = file_split[0] + file_split[1] + 'x4.png'
        #filename += 'x4.png'
        # print(filename)
        filename_LR = filename.split(".")[0]+"x2.png"
        path = os.path.join(args.inference_result, filename_LR)
        # print(path)
        #if not os.path.isfile(path):
        #    raise FileNotFoundError('')
        print("==================================")
        
        #处理图像对应问题
        if os.path.exists(path):
            inf_img = cv2.imread(path).astype(np.float32) / 255
        
            if (hr_img.shape != inf_img.shape):
                hr_img = cv2.resize(hr_img, (inf_img.shape[1], inf_img.shape[0]),
                                    interpolation=cv2.INTER_CUBIC)

            if test_y_channel and hr_img.ndim == 3 and hr_img.shape[2] == 3:
                hr_img = bgr2ycbcr(hr_img, y_only=True)
                inf_img = bgr2ycbcr(inf_img, y_only=True)

            # compare HR image and inferenced image with measures
            print('-' * 10)
            if calc_psnr:
                psnr = calculate_psnr(hr_img * 255, inf_img * 255,
                                      args.crop_border)
                print('{0} : PSNR {1:.3f} dB'.format(filename, psnr))
                mean_psnr.append(psnr)
            if calc_ssim:
                ssim = calculate_ssim(hr_img * 255, inf_img * 255,
                                      args.crop_border)
                print('{0} : SSIM {1:.3f}'.format(filename, ssim))
                mean_ssim.append(ssim)

    print('-' * 10)
    if calc_psnr:
        print('mean-PSNR {:.3f} dB'.format(sum(mean_psnr) / len(mean_psnr)))
    if calc_ssim:
        print('mean-SSIM {:.3f}'.format(sum(mean_ssim) / len(mean_ssim)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--HR_data_dir',
        default='./DIV2K_test_100/DIV2K_train_HR_801_900',
        type=str)
    parser.add_argument('--inference_result',
                        default='./DIV2K_test_predicted/DIV2K_train_LR_bicubic_801_900_X2/',
                        type=str)
    parser.add_argument('--crop_border', default=4, type=int)
    args = parser.parse_args()

    calc_measures(args, calc_psnr=True, calc_ssim=True)

