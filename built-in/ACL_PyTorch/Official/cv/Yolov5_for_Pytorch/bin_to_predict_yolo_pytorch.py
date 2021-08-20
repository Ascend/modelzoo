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

import os
import numpy as np
import argparse
import time
import torch
import torchvision
import cv2
import json
from tqdm import tqdm


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float)


def sigmoid(x0):
    s = 1 / (1 + np.exp(-x0))
    return s

def detect(x, model_type):
    """
    x(bs,3,20,20,85)
    """

    # x(bs,3,20,20,85)
    z = []
    grid = []
    for i in range(3):
        _, _, ny, nx, _ =  x[i].shape
        grid.append(_make_grid(nx, ny))
    if model_type == 'yolov5':
        stride =  np.array([8, 16, 32])
        anchor_grid = np.array(
            [[10., 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])\
            .reshape(3, 1, 3, 1, 1, 2)
    elif model_type == 'yolov3':
        stride = np.array([32, 16, 8])
        anchor_grid = np.array(
            [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10., 13, 16, 30, 33, 23]])\
            .reshape(3, 1, 3, 1, 1, 2)

    for i in range(3):
        y = sigmoid(x[i])
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.reshape(1, -1, 85))
    return np.concatenate(z, 1)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def post_process(flags):
    names = np.loadtxt(flags.coco_class_names, dtype='str', delimiter='\n')
    img = torch.zeros((1, 3, flags.net_input_size, flags.net_input_size))

    # 读取bin文件用于生成预测结果
    bin_path = flags.bin_data_path
    ori_path = flags.origin_jpg_path

    det_results_path = flags.det_results_path
    os.makedirs(det_results_path, exist_ok=True)
    total_img = set([name[:name.rfind('_')]
                     for name in os.listdir(bin_path) if "bin" in name])
    coco91class = coco80_to_coco91_class()
    jdict = []
    for bin_file in tqdm(sorted(total_img)):
        path_base = os.path.join(bin_path, bin_file)
        src_img = cv2.imread(os.path.join(ori_path, '{}.jpg'.format(bin_file)))
        assert src_img is not None, 'Image Not Found ' + bin_file

        # 加载检测的所有输出tensor
        res_buff = []

        if flags.model_type == 'yolov5':
            # yolo_shape = [[1, 3, 85, 80, 80], [1, 3, 85, 40, 40], [1, 3, 85, 20, 20]]
            yolo_shape = [[1, 3, 80, 80, 85], [1, 3, 40, 40, 85], [1, 3, 20, 20, 85]]
        elif flags.model_type == 'yolov3':
            yolo_shape = [[1, 3, 85, 13, 13], [1, 3, 85, 26, 26], [1, 3, 85, 52, 52]]

        for num in range(flags.net_out_num):
            # print(path_base + "_" + str(num + 1) + ".bin")
            if os.path.exists(path_base + "_" + str(num + 1) + ".bin"):
                buf = np.fromfile(path_base + "_" +
                                  # str(num + 1) + ".bin", dtype="float32")
                                  str(num + 1) + ".bin", dtype="float16").astype(np.float32)
                # res_buff.append(buf.reshape(yolo_shape[num]).transpose((0, 1, 3, 4, 2))) # 1,3,85,h,w ->1,3,h,w,85
                res_buff.append(buf.reshape(yolo_shape[num]))
            else:
                print("[ERROR] file not exist", path_base +
                      "_" + str(num + 1) + ".bin")

        res_tensor = detect(res_buff, flags.model_type)
        res_tensor = torch.from_numpy(res_tensor)
        # Apply NMS
        # pred = non_max_suppression(res_tensor, conf_thres=0.33, iou_thres=0.5, classes=None, agnostic=False)
        pred = non_max_suppression(res_tensor, conf_thres=flags.conf_thres, iou_thres=flags.iou_thres, classes=None, agnostic=False)
        det_results_file = os.path.join(det_results_path, bin_file + ".txt")

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None:
                predn = det.clone()
                scale_coords(img.shape[2:], predn[:, :4], src_img.shape)  # native-space pred
                image_id = int(bin_file.split('_')[-1])  # format: xx_imgid.bin, need modify when use other format
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(det.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            if flags.save_txt:
                size  =''
                size += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(src_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # Rescale boxes from img_size to im0 size
                if det is not None:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], src_img.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        size += '%g %ss, ' % (n, names[int(c)])  # add to string
                    with open(det_results_file, 'w') as f:
                        for *xyxy, conf, cls in det:
                            content = '{} {} {} {} {} {}'.format(names[int(cls)], conf, *xyxy)
                            f.write(content)
                            f.write('\n')
                else:
                    with open(det_results_file, 'w') as f:
                        f.write('')

    pred_json = "predictions.json"
    print('\nsaving %s...' % pred_json)
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)
    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--origin_jpg_path", default="./val2017/")
    parser.add_argument("--save_txt", action="store_true", default=False)
    parser.add_argument("--det_results_path",
                        default="./detection-results/")
    parser.add_argument("--coco_class_names", default="./coco2017.names")
    parser.add_argument("--net_input_size", default=640, type=int)
    parser.add_argument("--net_out_num", default=3)
    parser.add_argument("--model_type", default='yolov5')
    parser.add_argument("--conf_thres", type=float, default=0.001)
    parser.add_argument("--iou_thres", type=float, default=0.65)
    flags = parser.parse_args()

    post_process(flags)
