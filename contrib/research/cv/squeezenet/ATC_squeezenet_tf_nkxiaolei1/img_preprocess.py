import os
import cv2
import numpy as np
import sys

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def center_crop(img, out_height, out_width):
    height, width = img.shape[:2]
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def pre_process_img(img, dims=None, precision="fp32"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_LINEAR
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    STDDEV_RGB = [1.0 * 128, 1.0 * 128, 1.0 * 128]

    if precision=="fp32":
        img = np.asarray(img, dtype='float32')
    if precision=="fp16":
        img = np.asarray(img, dtype='float16')
    
    mean = np.array([104.006,116.669,122.679],dtype=np.float32)
    img -= mean
    return img

if __name__ == "__main__":
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    files = os.listdir(src_path)
    files.sort()
    img_zise = '227,227,3'
    image_size = list(map(int, img_zise.split(",")))
    for file in files:
        if file.endswith('.JPEG'):
            src = src_path + "/" + file
            print("start to process %s"%src)
            img_org = cv2.imread(src)
            res = pre_process_img(img_org,dims=image_size,precision ='fp32')
            res.tofile(dst_path+"/" + file+".bin")
