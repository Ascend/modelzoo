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
import cv2
import numpy as np
import sys

src_path = sys.argv[1]
dst_path = sys.argv[2]

images = os.listdir(src_path)
images.sort()

for image_path in images:
    print("Start process image: {}".format(image_path))
    original_image = cv2.imread(os.path.join(src_path,image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image,(416,416))
    image_data = image_data / 255.
    image_data = image_data.astype("float32")
    image_data.tofile(os.path.join(dst_path,image_path[:-4]+".bin"))
