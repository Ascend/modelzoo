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
import sys

image_PATH = sys.argv[1]
result_PATH = sys.argv[2]
all_image_NAME = os.listdir(image_PATH)
all_image_NAME.sort()
scale_txt = "./retinanet_postprocess/rawScale.txt"
content = ""

cnter = 0
print("-------->>  %i images to be handled" %(len(all_image_NAME)))
for OneImageName in all_image_NAME:
    print("---> %i images has been processed" %cnter)
    image = cv2.imread(image_PATH+OneImageName)
    h,w,_ = image.shape
    content += "{} {} {}\n".format(OneImageName,1024/w,768/h)
    image = cv2.resize(image, (1024, 768), interpolation = cv2.INTER_AREA)
    image = image.astype(np.float32)
    image -= [103.939, 116.779, 123.68]
    image.tofile(result_PATH+OneImageName.split(".")[0]+".bin")
    cnter+=1

with open(scale_txt,'w') as f:
    f.write(content)

print("THE PROGRAM ENDED SUCCESSFULLY: %i" %cnter)
