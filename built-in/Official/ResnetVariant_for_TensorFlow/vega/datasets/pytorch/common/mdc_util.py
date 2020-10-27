# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This script is used to process the MDC dataset."""
import json
import numpy as np


def read_anno_VP(anno_info):
    """Read the annotation.

    if the dataset in ['Vehicle_person_anno', 'Vehicle_anno', 'Person_anno'], this function will be used.

    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["Vehicle_and_Person"]['list']
    if 'size' in anno["Vehicle_and_Person"]:
        img_width = anno["Vehicle_and_Person"]['size']['width']
        img_height = anno["Vehicle_and_Person"]['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["type"]
        truncated = obj["truncated"]
        occluded = obj["occluded"]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        if label_map[name] not in vehicle_person_id:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(int(bool(occluded) or bool(truncated)))
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno_Traff(anno_info):
    """Read the annotation.

    if the dataset in['Vehicle_person_mcl', 'Trafficsign_mcl', 'Trafficlight_shape_mcl', 'Traffcstick_mcl',
                        'Vehicle_mcl', 'Person_mcl'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["object"]
    if 'size' in anno:
        img_width = anno['size']['width']
        img_height = anno['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["name"]
        truncated = obj["truncated"]
        occluded = obj["occluded"]
        ignore = obj['ignore'] if 'ignore' in obj else 0
        difficult = obj['difficult'] if 'difficult' in obj else 0
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        if label_map[name] not in vehicle_person_id:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(int(bool(occluded) or bool(truncated) or bool(ignore) or bool(difficult)))
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno_Traff_mcl(anno_info):
    """Read the annotation.

    if the dataset in ['Trafficlight_mcl'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    label_tl_map = ['Unknow', 'Red', 'Yellow', 'Green', 'Back', 'TrafficLight_Red', 'TrafficLight_Yellow',
                    'TrafficLight_Green', 'TrafficLight_Black']
    anno = json.loads(open(anno_path).read())
    objs = anno["object"]
    if 'size' in anno:
        img_width = anno['size']['width']
        img_height = anno['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["name"]
        if name not in label_tl_map:
            continue
        if 'ignore' in obj:
            if obj['ignore'] == 1 and name == 'Black':
                continue
        truncated = obj["truncated"]
        occluded = obj["occluded"]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(int(bool(occluded) or bool(truncated)))
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_annno_bicycle(anno_info):
    """Read the annotation.

    if the dataset in  ['Bicycle_mcl'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["object"]
    if 'size' in anno:
        img_width = anno['size']['width']
        img_height = anno['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["name"]
        occluded = obj["occluded"]
        crowd = obj["Crowded"]
        ride = obj["Ride"]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if crowd > 0 or ride > 0:
            continue
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(occluded)
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno_bicycle_anno(anno_info):
    """Read the annotation.

    if the dataset in  ['Bicycle_anno'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["Bicycle"]['list']
    if 'size' in anno["Bicycle"]:
        img_width = anno["Bicycle"]['size']['width']
        img_height = anno["Bicycle"]['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["type"]
        occluded = obj["occluded"]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(occluded)
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno_Traff_anno(anno_info):
    """Read the annotation.

    if the dataset in ['Traffcstick_anno'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["TrafficStick"]['list']
    if 'size' in anno["TrafficStick"]:
        img_width = anno["TrafficStick"]['size']['width']
        img_height = anno["TrafficStick"]['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["type"]
        occluded = obj["occluded"]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(occluded)
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno_Trafficsign_anno(anno_info):
    """Read the annotation.

    if the dataset in ['Trafficsign_anno'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["TrafficSign"]['list']
    if 'size' in anno["TrafficSign"]:
        img_width = anno["TrafficSign"]['size']['width']
        img_height = anno["TrafficSign"]['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["bndbox"]
        name = obj["type"]
        if name == 'UnclearSign':
            continue
        occluded = obj["occluded"]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_ignore:
            is_crowd.append(occluded)
        elif set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno_Trafficlight_anno(anno_info):
    """Read the annotation.

    if the dataset in ['Trafficlight_anno'], this function will be used.
    :return: boxes, klass, is_crowd
    :rtype: tuple
    """
    (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd) = anno_info
    anno = json.loads(open(anno_path).read())
    objs = anno["TrafficLight"]['list']
    if 'size' in anno["TrafficLight"]:
        img_width = anno["TrafficLight"]['size']['width']
        img_height = anno["TrafficLight"]['size']['height']
    else:
        img_width = 1920
        img_height = 1080
    for obj in objs:
        bndbox = obj["outbox"]['bndbox']
        inboxes = obj["inboxList"]
        if len(inboxes) != 1:
            name = 'Back'
        else:
            if inboxes[0]['shape'] == -1:
                name = 'Back'
            else:
                name = tl_color_map[int(inboxes[0]['color'])]
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        fake = obj['Fake'] if 'Fake' in obj else 0
        if xmin >= xmax or ymin >= ymax:
            continue
        if (xmin < 1 and xmax < 1) or (ymin < 1 and ymax < 1) or (xmin >= img_width and xmax >= img_width) or (
                ymin >= img_height and ymax >= img_height):
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        klass.append(label_map[name])
        if set_fake:
            is_crowd.append(int(fake))
        else:
            is_crowd.append(0)
    return boxes, klass, is_crowd


def read_anno(anno_path, dataset, set_ignore=False, set_fake=False):
    """Read annotation of different format and change to the same format for training.

    :param anno_path: the annotation json file
    :type anno_path: str
    :param dataset: the dataset name of the annoation
    :type dataset: str
    :param set_ignore: True then set is_crowd according to truncated/occlude/difficult/ignore, defaults to False
    :type set_ignore: bool, optional
    :param set_fake: True then set is_crowd according to Fake key, defaults to False
    :type set_fake: bool, optional
    :return: boxes, n*4, [xmin, ymin, xmax, ymax] for each row
              klass, n
              is_crowd, n, 1/0 for crowded/non-crowded
              task_labels,n
    :rtype: list
    """
    if '\r\n' in anno_path:
        anno_path = anno_path.split('\r\n')[0]
    elif '\r' in anno_path:
        anno_path = anno_path.split('\r')[0]
    elif '\n' in anno_path:
        anno_path = anno_path.split('\n')[0]
    else:
        pass
    label_map = {'Pedestrian': 1, 'Cyclist': 2, 'Car': 3, 'Truck': 4, 'Machineshop': 4, 'Tram': 5,
                 'Motocycle': 8, 'Bicycle': 7, 'Pedicab': 7, 'pick_up': 3, 'Tricycle': 6, 'Misc': 6,
                 'tricycle-express': 7, 'TrafficLight': 9, 'TrafficSign': 10,
                 'RoadSign': 11, 'TrafficCone': 12, 'TrafficStick': 13, 'Billboard': 14, 'ParkingSlot': 15,
                 'FireHydrant': 16,
                 'Red': 17, 'Yellow': 18, 'Green': 19, 'Back': 20, 'Black': 20, 'Black': 20, 'UnclearSign': 10,
                 'TrafficLight_Red': 17, 'TrafficLight_Yellow': 18, 'TrafficLight_Green': 19, 'TrafficLight_Black': 20,
                 'TL_Circle': 21, 'TL_Up': 22, 'TL_Left': 23, 'TL_Right': 24, 'GuideSign': 25, 'BarrierGate': 26,
                 'pedestrian': 1, 'cyclist': 2, 'car': 3, 'truck': 4, 'machineshop': 4, 'tram': 5,
                 'motocycle': 8, 'bicycle': 7, 'pedicab': 7, 'tricycle': 6, 'misc': 6, 'CementTruck': 4
                 }
    tl_color_map = {0: 'Red', 1: 'Yellow', 2: 'Green'}
    vehicle_person_id = [1, 2, 3, 4, 5, 6]
    boxes = []
    klass = []
    is_crowd = []
    anno_info = (anno_path, set_ignore, set_fake, label_map, tl_color_map, vehicle_person_id, boxes, klass, is_crowd)
    return choose_anno(dataset, anno_info)


def choose_anno(dataset, anno_info):
    """Choose different function to read annotation."""
    if dataset in ['Vehicle_person_anno', 'Vehicle_anno', 'Person_anno']:
        boxes, klass, is_crowd = read_anno_VP(anno_info)
    elif dataset in ['Vehicle_person_mcl', 'Trafficsign_mcl', 'Trafficlight_shape_mcl', 'Traffcstick_mcl',
                     'Vehicle_mcl', 'Person_mcl']:
        boxes, klass, is_crowd = read_anno_Traff(anno_info)
    elif dataset in ['Trafficlight_mcl']:
        boxes, klass, is_crowd = read_anno_Traff_mcl(anno_info)
    elif dataset in ['Bicycle_mcl']:
        boxes, klass, is_crowd = read_annno_bicycle(anno_info)
    elif dataset in ['Bicycle_anno']:
        boxes, klass, is_crowd = read_anno_bicycle_anno(anno_info)
    elif dataset in ['Traffcstick_anno']:
        boxes, klass, is_crowd = read_anno_Traff_anno(anno_info)
    elif dataset in ['Trafficsign_anno']:
        boxes, klass, is_crowd = read_anno_Trafficsign_anno(anno_info)
    elif dataset in ['Trafficlight_anno']:
        boxes, klass, is_crowd = read_anno_Trafficlight_anno(anno_info)
    else:
        raise 'err'
    boxes = np.asarray(boxes, dtype='float32')
    klass = np.asarray(klass, dtype='int64')
    is_crowd = np.asarray(is_crowd, dtype='bool')
    boxes_gt = boxes[~is_crowd]
    boxes_crowd = boxes[is_crowd]
    klass_gt = klass[~is_crowd]
    return boxes_gt, klass_gt, boxes_crowd
