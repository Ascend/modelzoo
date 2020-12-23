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
import array
import collections
import json
import os
import sys
import threading
import time
from queue import Queue
import cv2
import numpy as np
import math
from icdar import restore_rectangle
import lanms

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./img_input", help="path to the dataset")
    parser.add_argument("--output_path", default="./img_output", help="path to the dataset")
    parser.add_argument("--backend", default="acl", help="runtime to use")
    parser.add_argument("--model", required=True, help="model file path")
    parser.add_argument("--inputs", default="input:0", help="model inputs nodes eg: data1:0 ")
    parser.add_argument("--outputs", default="softmax:0", help="model outputs nodes list eg:fc1:0,fc2:0,fc3:0 ")

    # below will override DNMetis rules compliant settings - don't use for official submission
    parser.add_argument("--count", default=1000, type=int, help="dataset items to infer")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8", "uint8"],
                        help="precision mode, one of " + str(["fp32", "fp16", "int8", "uint8"]))
    parser.add_argument("--feed", default=[], help="feed")
    parser.add_argument("--image_list", default=[], help="image_list")
    parser.add_argument("--label_list", default=[], help="label_list")
    parser.add_argument("--cfg_path", default="./backend_cfg/built-in_config.txt")

    args = parser.parse_args()
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")
    return args

def get_backend(backend):
    if backend == "acl":
        from backend.backend_acl import AclBackend
        backend = AclBackend()
    return backend

def resize_image(im, max_side_len=2400, resize_h=768, resize_w=768):
    '''
    因为NPU不支持动态shape，所以这里不能简单的根据模型的输入的shape 768*768，
    直接进行resize，
    否则会因为图像的失真而导致精度的下降
    '''
    h, w, _ = im.shape
    print("origin_h:%d, resize_h:%d"%(h,resize_h))
    print("origin_w:%d, resize_w:%d"%(w,resize_w))

    if h <= resize_h and w <= resize_w:
        im = cv2.copyMakeBorder(im,0,resize_h-h,0,resize_w-w,cv2.BORDER_CONSTANT,value=(0,0,0))
        ratio_h = 1
        ratio_w = 1
    else:
        ratio_w = ratio_h = resize_h/max(h,w)
        im = cv2.resize(im, (math.floor(w*ratio_w), math.floor(h*ratio_h)))
        im = cv2.copyMakeBorder(im, 0, resize_h - math.floor(h*ratio_h), 0, resize_w - math.floor(w*ratio_w), cv2.BORDER_CONSTANT, value=(0, 0, 0))

    print("ratio_h=ratio_w=%.4f"%ratio_h)
    im = np.asarray(im, dtype='float32')
    return im, (ratio_h, ratio_w)

def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def main():
    #args
    args = get_args()

    # find backend
    backend = get_backend(args.backend)

    # load model to backend
    model = backend.load(args)

    #start:
    src_path = args.input_path
    files = os.listdir(src_path)
    files.sort()
    for file in files:
        if file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.png'):
            src = src_path + "/" + file
            #Pictures preprocess
            print("start to preprocess %s" % src)
            img_org = cv2.imread(src)[:, :, ::-1]
            im_resized, (ratio_h, ratio_w) = resize_image(img_org)

            #offline inference
            predictions = backend.predict(im_resized)
            score = predictions[0]
            geometry = predictions[1]

            #Pictures postprocess
            print("start to postprocess %s" % src)
            timer = {'net': 0, 'restore': 0, 'nms': 0}
            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

                #save to pictures
                res_file = os.path.join(
                    args.output_path,
                    '{}.txt'.format(
                        os.path.basename(file).split('.')[0]))

                with open(res_file, 'w') as f:
                    for box in boxes:
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                        f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                        ))
                        cv2.polylines(img_org[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                img_path = os.path.join(args.output_path, file)
                cv2.imwrite(img_path, img_org[:, :, ::-1])
    #end
    backend.unload()

if __name__ == "__main__":
    main()
