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
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="ctpn prepare data")

parser.add_argument("--image_path", type=str, default="../image", help="The path of the ori image")
parser.add_argument("--output_path", type=str, default="../datasets", help="The path of the image bin")
parser.add_argument("--img_conf", type=str, default="./img_info", help="The path of the image config")
parser.add_argument("--crop_width", type=int, default="", help="Width of the image cropping")
parser.add_argument("--crop_height", type=int, default="", help="height of the image cropping")

args = parser.parse_args()

def get_reasoning_data(image_path):
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'JPEG']
    for parent, dirnames, filenames in os.walk(os.path.join(image_path)):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent,filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files

def image_resize(image):
    img_src = cv2.imread(image)[:, :, ::-1]
    img_size = img_src.shape
    new_height = args.crop_height
    new_width = args.crop_width
    rw = new_width/img_size[1]
    rh = new_height/img_size[0]
    re_im = cv2.resize(img_src,(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    h, w, c = re_im.shape
    img = np.array(re_im, dtype='float32')
    img_info = np.array([h, w, c])
    return img, img_info, h, w, c, rh, rw

def main(img_path, out_path, img_conf_path):
    img_list = get_reasoning_data(img_path)
    f = open(img_conf_path, "w+")
    i = 0
    for im_fn in img_list:
        image, image_info, h, w, c, rh, rw = image_resize(im_fn)
        image_name = im_fn.split('/')[-1].split('.')[0]
        bin_img = out_path + "/" + image_name + ".bin"
        image.tofile(bin_img)
        f.write(str(i) + " " + image_name  + " " + str(h) + " " + str(w) + " " + str(c) + " " + str(rh) + " " + str(rw))
        f.write('\n')
        i+=1
    f.close()

if __name__== '__main__':
    image_path = args.image_path
    output_path = args.output_path
    img_conf_path = args.img_conf
    main(image_path, output_path, img_conf_path)
