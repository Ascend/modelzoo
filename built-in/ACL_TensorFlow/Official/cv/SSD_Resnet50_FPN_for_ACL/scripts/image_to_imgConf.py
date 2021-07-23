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

import numpy as np
import argparse
import random
import os
from PIL import Image
import PIL

parser = argparse.ArgumentParser(description="SSD_mobilenet test single image test procedure.")
parser.add_argument("--save_conf_path", type=str, default="./img_info",
                    help="The path of the result.json.")
parser.add_argument("--intput_file_path", type=str, default="./acl/data",
                    help="The path of inference input bin file.")
args = parser.parse_args()
def get_reasoning_data(image_path):
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG','JPEG']
    for parent, dirnames, filenames in os.walk(os.path.join(image_path)):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files

def main():
    img_info_path = args.save_conf_path
    img_path = args.intput_file_path
    img_list = get_reasoning_data(img_path)
    f = open(img_info_path, "w+")
    i = 0
    for img_fn in img_list:
        try:
            img_name = img_fn.split("/")[-1].split(".")[0]
            img_src =  Image.open(img_fn)
            im_width ,im_height = img_src.size
            f.write(str(i) + " " + img_name + " " + str(im_width) + " " + str(im_height) )
            f.write('\n')

        except:
            print("Error reading image {}!".format(im_fn))
            continue
        i = i + 1
    f.close()

if __name__ == '__main__':
    main()

