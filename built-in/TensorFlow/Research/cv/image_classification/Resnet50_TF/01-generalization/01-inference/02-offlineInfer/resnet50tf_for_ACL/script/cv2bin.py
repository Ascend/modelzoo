import os
import sys
import cv2
import numpy as np
from PIL import Image

image_root = r'./image-50000'
resize_min = 256

def trans():
    images = os.listdir(image_root)
    for image_name in images:
        if image_name.endswith("txt"):
            continue
        # image_name = "20180522135150.jpg"
        print("the image name is {}....".format(image_name))
        image_path = os.path.join(image_root, image_name)
        #    image = read_image(image_path, 1)
        # img = Image.open(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        '''
        img = img.astype(np.float32)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        '''
        # height, width = img.size
        height = img.shape[1]
        width = img.shape[0]
        # print('=====',height,width)
        smaller_dim = np.minimum(height, width)
        scale_ratio = resize_min / smaller_dim
        new_height = int(height * scale_ratio)
        new_width = int(width * scale_ratio)
        # img = img.resize((new_height, new_width)) ##

        img = cv2.resize(img, (new_height, new_width))

        img = np.array(img)
        if len(img.shape) != 3:
            continue
        height, width, c = img.shape
        crop_height = crop_width = 224
        amount_to_be_cropped_h = (height - crop_height)
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = (width - crop_width)
        crop_left = amount_to_be_cropped_w // 2
        img = img[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]  ##[y0:y1,x0:x1]

        #img = np.array(img,dtype=np.float32)[np.newaxis, :, :, :]
        # means = [103.939, 116.779,123.68 ]
        means = [123.68, 116.779, 103.939]
        #means = np.array(means, dtype=np.float32)
        img = img - means  
        img = np.array(img,dtype=np.float32)[np.newaxis, :, :, :]
        #print('===',img.shape)
        #print('===',img.size)


        img.tofile('./cv2bin-50000-xzm/{}.bin'.format(image_name))
        # print(image)
        # print(image.dtype)
        # print(image.shape)
trans()
