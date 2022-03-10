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

import numpy as np

import skimage
from skimage import transform
from PIL import Image

from constants import scale_fact


def float_im(img):
    return np.divide(img, 255.)


# Adapted from: https://stackoverflow.com/a/39382475/9768291
def crop_center(img, crop_x, crop_y):
    """
    To crop the center of an image
    :param img: the image
    :param crop_x: how much to crop on the x-axis
    :param crop_y: how much to crop on the y-axis
    :return: cropped image, floated (values between 0 and 1)
    """
    y, x, _ = img.shape
    start_x = x//2-(crop_x // 2)
    start_y = y//2-(crop_y // 2)

    cropped_img = img[start_y:start_y + crop_y, start_x:start_x + crop_x]
    return float_im(cropped_img)


# TODO: provide some way of saving FLOAT images
def save_np_img(np_img, path, name):
    """
    To save the image.
    :param np_img: numpy_array type image
    :param path: string type of the existing path where to save the image
    :param name: string type that includes the format (ex:"bob.png")
    :return: numpy array
    """

    assert isinstance(path, str), 'Path of wrong type! (Must be String)'
    assert isinstance(name, str), 'Name of wrong type! (Must be String)'

    # TODO: To transform float-arrays into int-arrays (see https://stackoverflow.com/questions/52490653/saving-float-numpy-images)
    if type(np_img[0][0][0].item()) != int:
        np_img = np.multiply(np_img, 255).astype(int)
        # File "C:\Users\payne\Anaconda3\envs\ml-gpu\lib\site-packages\PIL\Image.py", line 2460, in fromarray
        #     mode, rawmode = _fromarray_typemap[typekey]
        # KeyError: ((1, 1, 3), '<i4')
        # File  "C:\Users\payne\Anaconda3\envs\ml-gpu\lib\site-packages\PIL\Image.py", line 2463, in fromarray
        #     raise TypeError("Cannot handle this data type")
        # TypeError: Cannot handle this data type

    im = Image.fromarray(np_img)
    im.save(path + name)
    return np_img


def single_downscale(img, width, height):
    """
    Downscales an image by the factor set in the 'constants'
    :param img: the image, as a Numpy Array
    :param width: width to be downscaled
    :param height: height to be downscaled
    :return: returns a float-type numpy by default (values between 0 and 1)
    """
    # TODO: look into `skimage.transform.downscale_local_mean()`
    scaled_img = skimage.transform.resize(
        img,
        (width // scale_fact, height // scale_fact),
        mode='reflect',
        anti_aliasing=True)
    return scaled_img