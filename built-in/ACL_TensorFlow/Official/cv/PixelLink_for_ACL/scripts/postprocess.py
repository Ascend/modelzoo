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
import math
import tensorflow as tf
import sys
import os
import subprocess as commands
import cv2

session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
)

def inferenceModel(pixel_pos_scores,link_pos_scores):
    softmax1 = tf.nn.softmax(pixel_pos_scores)
    pixel_pos_scores_result = tf.strided_slice(softmax1,begin=[0,0,0,1],end=[0,0,0,2],strides=[1,1,1,1],begin_mask=7,end_mask=7,ellipsis_mask=0,new_axis_mask=0,shrink_axis_mask=8)
    link_pos_scores_shape=np.reshape(link_pos_scores,(1,192,320,8,2))
    softmax2 = tf.nn.softmax(link_pos_scores_shape)
    link_pos_scores_result = tf.strided_slice(softmax2,begin=[0,0,0,0,1],end=[0,0,0,0,2],strides=[1,1,1,1,1],begin_mask=15,end_mask=15,ellipsis_mask=0,new_axis_mask=0,shrink_axis_mask=16)
    with tf.Session(config=session_config) as sess:
        a,b = sess.run([pixel_pos_scores_result,link_pos_scores_result])
    return a,b

def is_valid_cord(x,y,w,h):
    """
    Tell whether the 2D coordinate (x,y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y>=0 and y < h;

def get_neighbours_8(x,y):
    """
    Get 8 neighbours of point(x,y)
    """
    return [(x-1,y-1),(x,y-1),(x+1,y-1), \
            (x-1,y),(x+1,y), \
            (x-1,y+1),(x,y+1),(x+1,y+1)]

def decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = zip(*np.where(pixel_mask))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)
    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True
        if update_parent:
            set_parent(point,root)

        return root
    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2:
            set_parent(root1,root2)

    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        mask = np.zeros_like(pixel_mask, dtype=np.int32)
        for point in zip(*np.where(pixel_mask)):
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    for point in zip(*np.where(pixel_mask)):
        y,x = point
        neighbours = get_neighbours_8(x,y)
        for n_idx,(nx,ny) in enumerate(neighbours):
            if is_valid_cord(nx,ny,w,h):
                link_value = link_mask[y,x,n_idx]
                pixel_cls = pixel_mask[ny,nx]
                if link_value and pixel_cls:
                    join(point,(ny,nx))

    mask = get_all()
    return mask

def decode_image(pixel_scores, link_scores,
                 pixel_conf_threshold,link_conf_threshold):
    mask = decode_image_by_join(pixel_scores, link_scores,pixel_conf_threshold,link_conf_threshold)
    return mask

def decode_batch(pixel_cls_scores, pixel_link_scores,
                 pixel_conf_threshold=None, link_conf_threshold=None):
    if pixel_conf_threshold is None:
        pixel_conf_threshold = 0.8

    if link_conf_threshold is None:
        link_conf_threshold = 0.8

    batch_size = 1
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx,:,:]
        image_pos_link_scores = pixel_link_scores[image_idx,:,:,:]
        mask = decode_image(
            image_pos_pixel_scores,image_pos_link_scores,
            pixel_conf_threshold,link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask,np.int32)

def find_contours(mask, method=None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype=np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                       method=method)
    except:
        _,contours,_ =cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                       method=method)
    return contours

def points_to_contour(points):
    contours = [[list(p)] for p in points]
    return np.asarray(contours, dtype = np.int32)

def min_area_rect(cnt):
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w*h

def rect_to_xys(rect, image_shape):
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w-1
        return x
    def get_valid_y(y):
        if y <0:
            return 0
        if y >=h:
            return h-1
        return y

    rect = ((rect[0], rect[1]),(rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy,(x,y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x,y]
    points = np.reshape(points, -1)
    return points

def mask_to_bboxes(mask, image_shape= None, min_area=None,
                  min_height=None, min_aspect_ratio=None):
    image_h,image_w = image_shape[0:2]
    if min_area is None:
        min_area = 300
    if min_height is None:
        min_height = 10
    bboxes = []
    max_bbox_idx = mask.max()
    mask = cv2.resize(mask, (image_w,image_h), interpolation=cv2.INTER_NEAREST)

    for bbox_idx in range(1, max_bbox_idx+1):
        bbox_mask = mask == bbox_idx
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        w,h = rect[2:-1]
        if min(w,h) < min_height:
            continue
        if rect_area < min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)

    return bboxes

def to_txt(txt_path, image_name,
           image_data, pixel_pos_scores, link_pos_scores):
    def write_result_as_txt(image_name, bboxes, path):
        filename = 'res_'+image_name+'.txt'
        lines = []
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
            lines.append(line)
        with open(os.path.join(txt_path,filename), 'w') as f:
            for line in lines:
                f.write(line)

    mask = decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]

    bboxes = mask_to_bboxes(mask, image_data)
    write_result_as_txt(image_name, bboxes, txt_path)

def cmd(cmd):
    print("Cmd is",cmd)
    return commands.getoutput(cmd)

def test(inputFolder, outputFolder, i ,image_data_shape, ZIPFolder):
    image_name = 'img_' + str(i)
    output0 = 'davinci_' + image_name + '_output0.bin'
    output1 = 'davinci_' + image_name + '_output1.bin'
    pixel_pos_scores = np.fromfile(os.path.join(inputFolder,output0),np.float32).reshape(1,192,320,2)
    link_pos_scores = np.fromfile(os.path.join(inputFolder,output1),np.float32).reshape(1,192,320,16)
    pixel_pos_scores,link_pos_scores = inferenceModel(pixel_pos_scores,link_pos_scores)
    to_txt(outputFolder,image_name,image_data_shape, pixel_pos_scores,link_pos_scores)

if __name__ == '__main__':
    inputFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    ZIPFolder = os.path.join(outputFolder,"result.zip")
    image_data_shape = (720,1280,3)
    for i in range(1,501):
        test(inputFolder, outputFolder, i ,image_data_shape, ZIPFolder)
    cmd = 'zip -j %s %s/*'%(ZIPFolder, outputFolder);
    os.system(cmd);
    print("zip is done!")



