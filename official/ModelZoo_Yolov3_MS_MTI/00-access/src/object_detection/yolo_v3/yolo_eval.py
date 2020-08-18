# -*- coding: UTF-8 -*-

import numpy as np
from src.object_detection.yolo_v3.config import Config_yolov3


def NMS(class_boxes, class_box_scores, max_boxes, nms_threshold):
    x1 = class_boxes[:, 1:2]
    y1 = class_boxes[:, 0:1]
    x2 = class_boxes[:, 3:4]
    y2 = class_boxes[:, 2:3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = class_box_scores.argsort(axis=0)[::-1]
    order = np.squeeze(order)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        # calculate interaction
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h

        # calculate one vs rest IOU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]

    if len(keep) > max_boxes:
        keep = keep[:max_boxes]
    return keep


def yolo_eval(yolo_outputs, image_shape, max_boxes=50):
    """
    Introduction
    ------------
        根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
    Parameters
    ----------
        yolo_outputs: yolov3_darknet53 模块的输出(非训练模型)
        image_shape: 图片的大小
        max_boxes:  最大box数量
    Returns
    -------
        boxes_: 物体框的位置
        scores_: 物体类别的概率
        classes_: 物体类别
    单张图片做eval
    """
    def _yolo_boxes_scores(box_xy, box_wh, box_confidence, box_class_probs, num_classes, input_shape, image_shape):
        """
        Introduction
        ------------
            该函数是将box的坐标修正，除去之前按照长宽比缩放填充的部分，最后将box的坐标还原成相对原始图片的
        Parameters
        ----------
            feats: 模型输出feature map
            anchors: 模型anchors
            num_classes: 数据集类别数
            input_shape: 训练输入图片大小
            image_shape: 原始图片的大小

        输入时单张图片、单个layer_output
        注意：这里的输出 yx hw
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        image_shape = np.array(image_shape)[..., ::-1]
        input_shape = np.array(input_shape)

        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.

        boxes = np.concatenate((box_min[..., 0:1],
                                box_min[..., 1:2],
                                box_max[..., 0:1],
                                box_max[..., 1:2]),
                               axis=-1)

        boxes *= np.concatenate((image_shape, image_shape), axis=-1)
        boxes = boxes.reshape((-1, 4))
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = boxes_scores.reshape((-1, num_classes))
        # boxes = [y, x, h, w]
        return boxes, boxes_scores


    # 单张图片的eval
    class_nums = Config_yolov3.num_classes
    boxes = []
    box_scores = []
    # yolo_outputs[0] = 13*13  -> 13*32=416
    input_shape = np.array(np.shape(yolo_outputs[0])[1:3]) * 32
    # 从单张图分别取三个尺度, 最小的尺度在最前
    for i in range(len(yolo_outputs)):
        box_xy = yolo_outputs[i][..., :2]
        box_wh = yolo_outputs[i][..., :2:4]
        box_confidence = yolo_outputs[i][..., 4:5]
        box_class_probs = yolo_outputs[i][..., 5:]
        _boxes, _box_scores = _yolo_boxes_scores(box_xy=box_xy,
                                                 box_wh=box_wh,
                                                 box_confidence=box_confidence,
                                                 box_class_probs=box_class_probs,
                                                 num_classes=class_nums,
                                                 input_shape=input_shape,
                                                 image_shape=image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # 汇聚了三层的结果
    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    mask = box_scores >= Config_yolov3.obj_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    # 每种类别单独算NMS
    # 最终的输出需要经过两重筛选：1 obj_threshold 剔除背景； 2 NMS 剔除重叠
    for c in range(class_nums):
        class_boxes = boxes[mask[:, c]]
        class_box_scores = box_scores[mask[:, c]][:, c:c+1]
        nms_index = NMS(class_boxes, class_box_scores, max_boxes, Config_yolov3.nms_threshold)

        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]

        classes = np.ones_like(class_box_scores) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)

    # return box is [y x h w]
    return boxes_, scores_, classes_

