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
import argparse


def read_class_names(class_file_name):
    '''
    loads class name from a file
    '''
    names = {}
    with open(class_file_name, 'r') as data:
        for id, name in enumerate(data):
            names[id] = name.strip('\n')
    return names


def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values
    (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou


def non_max_suppression(predictions_with_boxes,
                        confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array,
    first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: deciding if prediction is valid
    :param iou_threshold: deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for image_pred in predictions:
        shape = image_pred.shape
        # non_zero_idxs = np.nonzero(image_pred)
        temp = image_pred
        sum_t = np.sum(temp, axis=1)
        non_zero_idxs = (sum_t != 0)

        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result


def parse_line(line):
    '''解析输入的confg文件，文件每一行是序号 文件名 宽 高'''
    temp = line.split(" ")
    index = temp[1].split("/")[-1].split(".")[0]
    # 放置高宽（不是宽高）
    return index, (int(temp[3]), int(temp[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./JPEGImages/")
    parser.add_argument("--test_annotation",
                        default="input/benchmark_input.txt")
    parser.add_argument("--det_results_path",
                        default="input/detection-results/")
    parser.add_argument("--coco_class_names", default="./coco.names")
    parser.add_argument("--voc_class_names", default="./voc.names")
    parser.add_argument("--net_input_size", default=416)
    parser.add_argument("--net_out_num", default=1)
    flags = parser.parse_args()

    # 根据标注文件生成字典，用于查询分辨率
    # 加载输入图片的宽高信息
    img_size_dict = dict()
    with open(flags.test_annotation)as f:
        for line in f.readlines():
            temp = parse_line(line)
            img_size_dict[temp[0]] = temp[1]

    # 加载coco和voc的index->类别的映射关系
    coco_class_map = read_class_names(flags.coco_class_names)
    voc_class_map = read_class_names(flags.voc_class_names)
    coco_set = set(coco_class_map.values())
    voc_set = set(voc_class_map.values())

    # 读取bin文件用于生成预测结果
    bin_path = flags.bin_data_path
    net_input_size = flags.net_input_size

    det_results_path = flags.det_results_path
    os.makedirs(det_results_path, exist_ok=True)
    total_img = set([name.split("_")[0]
                     for name in os.listdir(bin_path) if "bin" in name])
    for bin_file in sorted(total_img):
        path_base = os.path.join(bin_path, bin_file)
        # 加载检测的所有输出tensor
        res_buff = []
        for num in range(flags.net_out_num):
            print(path_base + "_" + str(num + 1) + ".bin")
            if os.path.exists(path_base + "_" + str(num + 1) + ".bin"):
                buf = np.fromfile(path_base + "_" +
                                  str(num + 1) + ".bin", dtype="float32")
                res_buff.append(buf)
            else:
                print("[ERROR] file not exist", path_base +
                      "_" + str(num + 1) + ".bin")
        res_tensor = np.concatenate(res_buff, axis=0)
        # resize成1*？*85维度，1置信+4box+80类
        pred_bbox = res_tensor.reshape([1, -1, 85])

        img_index_str = bin_file.split("_")[0]
        bboxes = []
        if img_index_str in img_size_dict:
            bboxes = non_max_suppression(pred_bbox, 0.5, 0.4)
        else:
            print("[ERROR] input with no width height")
        det_results_str = ""

        # YOLO输出是COCO类别，包含了VOC的20类，将VOC20类写入预测结果，用于map评估
        current_img_size = img_size_dict[img_index_str]
        for class_ind in bboxes:
            class_name = coco_class_map[class_ind]
            score = bboxes[class_ind][0][1]
            bbox = bboxes[class_ind][0][0]
            if class_name in voc_set:
                # 对每个维度的坐标进行缩放，从输入尺寸，缩放到原尺寸
                hscale = current_img_size[0] / float(flags.net_input_size)
                wscale = current_img_size[1] / float(flags.net_input_size)
                det_results_str += "{} {} {} {} {} {}\n".format(
                    class_name, str(score), str(bbox[0] * wscale),
                    str(bbox[1] * hscale), str(bbox[2] * wscale),
                    str(bbox[3] * hscale))

        det_results_file = os.path.join(
            det_results_path, img_index_str + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)
