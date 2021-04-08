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
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os
from config import aspect_ratio, scales, IMG_H, IMG_W, BATCH_SIZE, CLASSES, K, IMG_PATH, XML_PATH



def generate_anchors(area, stride):
    nums_A = len(aspect_ratio) * len(scales)
    feature_h, feature_w = int(np.ceil(IMG_H/stride)), int(np.ceil(IMG_W/stride))
    anchors = np.zeros([feature_h, feature_w, nums_A, 4])
    for i in range(feature_h):
        for j in range(feature_w):
            for k in range(nums_A):
                anchors[i, j, k, 0] = j * stride
                anchors[i, j, k, 1] = i * stride
            count = 0
            for s in scales:
                for ar in aspect_ratio:
                    anchors[i, j, count, 2] = area * s * np.sqrt(ar)
                    anchors[i, j, count, 3] = area * s / np.sqrt(ar)
                    count += 1
                    pass
    anchors = np.reshape(anchors, [-1, 4])
    return np.float32(anchors)

def read_data(xml_path, img_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    names = []
    gtbboxes = np.zeros([len(objects), 4], dtype=np.int32)
    for idx, obj in enumerate(objects):
        names.append(obj.find("name").text)
        xmin = int(obj.find("bndbox").find("xmin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        gtbboxes[idx, 0] = (xmin + xmax)//2
        gtbboxes[idx, 1] = (ymin + ymax)//2
        gtbboxes[idx, 2] = xmax - xmin
        gtbboxes[idx, 3] = ymax - ymin
    img = np.array(Image.open(img_path))
    labels = np.zeros([len(objects)])
    for idx, name in enumerate(names):
        labels[idx] = CLASSES.index(name)
    return img, gtbboxes, labels

def resize_img_and_bbox(img, gt_bbox):
    h, w = img.shape[0], img.shape[1]
    new_y = IMG_H * gt_bbox[:, 1:2] / h
    new_x = IMG_W * gt_bbox[:, 0:1] / w
    new_h = IMG_H * gt_bbox[:, 3:4] / h
    new_w = IMG_W * gt_bbox[:, 2:3] / w
    new_img = np.array(Image.fromarray(img).resize([IMG_W, IMG_H]))
    new_bbox = np.concatenate((new_x, new_y, new_w, new_h), axis=1)
    return new_img, new_bbox

def cal_iou(anchors, gt_bboxes):
    anchors = anchors[np.newaxis, :, :]
    gt_bboxes = gt_bboxes[:, np.newaxis, :]
    a_x1 = anchors[:, :, 0] - anchors[:, :, 2] / 2
    a_x2 = anchors[:, :, 0] + anchors[:, :, 2] / 2
    a_y1 = anchors[:, :, 1] - anchors[:, :, 3] / 2
    a_y2 = anchors[:, :, 1] + anchors[:, :, 3] / 2
    g_x1 = gt_bboxes[:, :, 0] - gt_bboxes[:, :, 2] / 2
    g_x2 = gt_bboxes[:, :, 0] + gt_bboxes[:, :, 2] / 2
    g_y1 = gt_bboxes[:, :, 1] - gt_bboxes[:, :, 3] / 2
    g_y2 = gt_bboxes[:, :, 1] + gt_bboxes[:, :, 3] / 2
    inter_x1 = np.maximum(a_x1, g_x1)
    inter_x2 = np.minimum(a_x2, g_x2)
    inter_y1 = np.maximum(a_y1, g_y1)
    inter_y2 = np.minimum(a_y2, g_y2)
    inter_area = np.maximum(0., inter_x2 - inter_x1) * np.maximum(0., inter_y2 - inter_y1)
    union_area = anchors[:, :, 2] * anchors[:, :, 3] + gt_bboxes[:, :, 2] * gt_bboxes[:, :, 3] - inter_area
    ious = inter_area / union_area
    return ious

def bbox2offset(anchors, gt_bboxes):
    t_x = (gt_bboxes[:, 0:1] - anchors[:, 0:1]) / anchors[:, 2:3]
    t_y = (gt_bboxes[:, 1:2] - anchors[:, 1:2]) / anchors[:, 3:4]
    t_w = np.log(gt_bboxes[:, 2:3] / anchors[:, 2:3])
    t_h = np.log(gt_bboxes[:, 3:4] / anchors[:, 3:4])
    return np.concatenate((t_x, t_y, t_w, t_h), axis=1)

def offset2bbox(anchors, t_bbox):
    bbox_x = t_bbox[:, 0:1] * anchors[:, 2:3] + anchors[:, 0:1]
    bbox_y = t_bbox[:, 1:2] * anchors[:, 3:4] + anchors[:, 1:2]
    bbox_w = np.exp(t_bbox[:, 2:3]) * anchors[:, 2:3]
    bbox_h = np.exp(t_bbox[:, 3:4]) * anchors[:, 3:4]
    return np.concatenate((bbox_x, bbox_y, bbox_w, bbox_h), axis=1)

def show_img(img, bbox):
    x1 = bbox[0] - bbox[2] / 2
    x2 = bbox[0] + bbox[2] / 2
    y1 = bbox[1] - bbox[3] / 2
    y2 = bbox[1] + bbox[3] / 2
    Image.fromarray(img[int(y1):int(y2), int(x1):int(x2)]).show()
    pass

def generate_batch(anchors, gt_bboxes, gt_bboxes_labels):
    ious = cal_iou(anchors, gt_bboxes)
    max_ious = np.max(ious, axis=0)
    background_mask = np.zeros_like(max_ious)
    background_mask[max_ious < 0.4] = 1
    foreground_mask = np.zeros_like(max_ious)
    foreground_mask[max_ious >= 0.5] = 1
    argmax_idx = np.argmax(ious, axis=0)
    labels = gt_bboxes_labels[argmax_idx]
    # labels = labels * (1.0 - background_mask) # make the element of the background is 0
    bboxes = gt_bboxes[argmax_idx]
    maxiou_idx = np.argmax(ious, axis=1) # highest IOU index
    foreground_mask[maxiou_idx] = 1
    background_mask[maxiou_idx] = 0
    labels[maxiou_idx] = gt_bboxes_labels
    bboxes[maxiou_idx] = gt_bboxes
    t_bbox = bbox2offset(anchors, bboxes)
    valid_mask = foreground_mask + background_mask
    one_hot = np.float32(labels[:, np.newaxis] == np.array([range(K)])) * foreground_mask[:, np.newaxis]
    return foreground_mask, valid_mask, one_hot, t_bbox

def draw_bbox(img, bbox, text):
    #bbox: [x1, y1, x2, y2]
    h, w = img.shape[0], img.shape[1]
    x1 = np.maximum(0, bbox[0])
    y1 = np.maximum(0, bbox[1])
    x2 = np.minimum(w - 1, bbox[2])
    y2 = np.minimum(h - 1, bbox[3])
    img[y1:y2, x1, :] *= 0
    img[y1:y2, x1, 1] += np.ones([y2 - y1], dtype=np.uint8) * 255
    img[y1:y2, x2, :] *= 0
    img[y1:y2, x2, 1] += np.ones([y2 - y1], dtype=np.uint8) * 255
    img[y1, x1:x2, :] *= 0
    img[y1, x1:x2, 1] += np.ones([x2 - x1], dtype=np.uint8) * 255
    img[y2, x1:x2, :] *= 0
    img[y2, x1:x2, 1] += np.ones([x2 - x1], dtype=np.uint8) * 255

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    x = int(bbox[0])
    y = int(bbox[1])
    draw.text((x, y), text)
    drawed_img = np.array(img)
    return drawed_img

def recover_ImgAndBbox_scale(img, pre_bbox):
    h, w = img.shape[0], img.shape[1]
    y1 = pre_bbox[1] * h / IMG_H
    y2 = pre_bbox[3] * h / IMG_H
    x1 = pre_bbox[0] * w / IMG_W
    x2 = pre_bbox[2] * w / IMG_W
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def data_augmentation(img, gt_bboxes):
    if np.random.uniform(0., 1.) > 0.5:
        img = np.flip(img, axis=1)
        gt_bboxes = np.concatenate((IMG_W - 1 - gt_bboxes[:, 0:1], gt_bboxes[:, 1:2], gt_bboxes[:, 2:3], gt_bboxes[:, 3:4]), axis=1)
    return img, gt_bboxes

def read_batch_data(anchors):
    filenames = os.listdir(XML_PATH)
    nums_anchors = anchors.shape[0]
    IMGS = np.zeros([BATCH_SIZE, IMG_H, IMG_W, 3])
    FOREGROUND_MASKS = np.zeros([BATCH_SIZE, nums_anchors])
    VALID_MASKS = np.zeros([BATCH_SIZE, nums_anchors])
    LABELS = np.zeros([BATCH_SIZE, nums_anchors, K])
    TARGET_BBOXES = np.zeros([BATCH_SIZE, nums_anchors, 4])
    for i in range(BATCH_SIZE):
        rand_idx = np.random.randint(0, len(filenames), 1)[0]
        IMG, GTBBOX, LABEL = read_data(XML_PATH + filenames[rand_idx], IMG_PATH + filenames[rand_idx][:-3] + "jpg")
        IMG, GTBBOX = resize_img_and_bbox(IMG, GTBBOX)
        IMG, GTBBOX = data_augmentation(IMG, GTBBOX)
        FOREGROUND_MASK, VALID_MASK, LABEL, TARGET_BBOX = generate_batch(anchors, GTBBOX, LABEL)
        IMGS[i] = IMG
        FOREGROUND_MASKS[i] = FOREGROUND_MASK
        VALID_MASKS[i] = VALID_MASK
        LABELS[i] = LABEL
        TARGET_BBOXES[i] = TARGET_BBOX
    return IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES

def img2mat():
    # Just used for Google Colab
    import scipy.io as sio
    filenames = os.listdir(XML_PATH)
    nums = filenames.__len__()
    IMGS = np.zeros([nums, IMG_H, IMG_W, 3], dtype=np.uint8)
    gt_bboxes = []
    labels = []
    for idx, filename in enumerate(filenames):
        IMG, GTBBOX, LABEL = read_data(XML_PATH + filenames[idx], IMG_PATH + filenames[idx][:-3] + "jpg")
        IMG, GTBBOX = resize_img_and_bbox(IMG, GTBBOX)
        IMGS[idx] = IMG
        gt_bboxes.append(GTBBOX)
        labels.append(LABEL)
        if idx % 10 == 0:
            print(idx)
    sio.savemat("pascal.mat", {"images": IMGS, "bbox": np.array(gt_bboxes), "labels": np.array(labels)})

def read_colab_batch_data(anchors, data):
    #Just used for Google Colab
    nums_anchors = anchors.shape[0]
    IMGS = np.zeros([BATCH_SIZE, IMG_H, IMG_W, 3])
    FOREGROUND_MASKS = np.zeros([BATCH_SIZE, nums_anchors])
    VALID_MASKS = np.zeros([BATCH_SIZE, nums_anchors])
    LABELS = np.zeros([BATCH_SIZE, nums_anchors, K])
    TARGET_BBOXES = np.zeros([BATCH_SIZE, nums_anchors, 4])
    data_num = data["images"].shape[0]
    for i in range(BATCH_SIZE):
        rand_idx = np.random.randint(0, data_num, 1)[0]
        IMG, GTBBOX, LABEL = data["images"][rand_idx], data["bbox"][0, rand_idx], data["labels"][0, rand_idx]
        IMG, GTBBOX = data_augmentation(IMG, GTBBOX)
        FOREGROUND_MASK, VALID_MASK, LABEL, TARGET_BBOX = generate_batch(anchors, GTBBOX, LABEL)
        IMGS[i] = IMG
        FOREGROUND_MASKS[i] = FOREGROUND_MASK
        VALID_MASKS[i] = VALID_MASK
        LABELS[i] = LABEL
        TARGET_BBOXES[i] = TARGET_BBOX
    return IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES