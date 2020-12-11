# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import random
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
from tqdm import trange
import json
import os,time

'''
coco weight from official checked 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.629

'''
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    parser.add_argument('--batchsize', default=1,
                        help="""batchsize""")
    parser.add_argument("--annotation_txt", type=str, default='./data/coco2014_minival_test.txt',
                        help="The path of the input image. Or annotation label txt.")
    parser.add_argument("--label_file", type=str, default='./data/instances_val2014.json',
                        help="label_file.")
    parser.add_argument("--model_path", type=str, default='pb_model_tf/yolov3_tf.pb',
                        help="The path of the input image. Or annotation label txt.")
    parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--max_test", type=int, default=4954,
                        help="max step for test")
    parser.add_argument("--score_thresh", type=float, default=1e-3,
                        help="score_threshold for test")
    parser.add_argument("--nms_thresh", type=float, default=0.55,
                        help="iou_threshold for test")
    parser.add_argument("--max_boxes", type=int, default=100,
                        help="max_boxes for test")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                        help="The path of the class names.")
    parser.add_argument("--save_img", type=bool, default=False,
                        help="whether to save detected-result image")
    parser.add_argument("--save_json", type=bool, default=True,
                        help="whether to save detected-result cocolike json")
    parser.add_argument("--save_json_path", type=str, default="./result.json",
                        help="The path of the result.json.")
    parser.add_argument('--input_tensor_name', default = 'input:0',
                        help = """input_tensor_name""")
    parser.add_argument('--output_tensor_name_1', default='concat_9:0',
                        help="""output_tensor_name""")
    parser.add_argument('--output_tensor_name_2', default='mul_9:0',
                        help="""output_tensor_name""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

class Classifier(object):
    # set batch_size
    args = parse_args()
    batch_size = int(args.batchsize)

    def __init__(self):
        # --------------------------------------------------------------------------------
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 1）run on Ascend NPU
        custom_op.parameter_map["use_off_line"].b = True
        # 2）recommended use fp16 datatype to obtain better performance
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        # 3）disable remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # 4）set graph_run_mode=0，obtain better performance
        custom_op.parameter_map["graph_run_mode"].i = 0
        # --------------------------------------------------------------------------------
        # load model， set graph input nodes and output nodes
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor_1 = self.graph.get_tensor_by_name(args.output_tensor_name_1)
        self.output_tensor_2 = self.graph.get_tensor_by_name(args.output_tensor_name_2)
        # create session
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        load fronzen graph
        :param model_file:
        :return:
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch_data):
        """
        do infer
        :param batch_data:
        :return:
        """
        boxes_list = []
        scores_list = []
        total_time = 0
        i = 0
        for data in batch_data:
            t = time.time()
            boxes_, scores_ = self.sess.run([self.output_tensor_1,self.output_tensor_2], feed_dict={self.input_tensor: data})
            if i > 0:
                total_time = total_time + time.time() - t
            i = i + 1
            boxes_list.append(boxes_)
            scores_list.append(scores_)
        return np.array(boxes_list), np.array(scores_list), total_time

    def batch_process(self, image_data):
        """
        batch process
        :return:
        """
        # Get the batch information of the current input data, and automatically adjust the data to the fixed batch
        n_dim = image_data.shape[0]
        batch_size = self.batch_size

        # if data is not enough for the whole batch, you need to complete the data
        m = n_dim % batch_size
        if m < batch_size and m > 0:
            # The insufficient part shall be filled with 0 according to n dimension
            pad = np.zeros((batch_size - m, 416, 416, 3)).astype(np.float32)
            image_data = np.concatenate((image_data, pad), axis=0)

        # Define the Minis that can be divided into several batches
        mini_batch = []
        i = 0
        while i < n_dim:
            # Define the Minis that can be divided into several batches
            mini_batch.append(image_data[i: i + batch_size, :, :, :])
            i += batch_size

        return mini_batch

def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors

def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    return image_padded, resize_ratio, dw, dh

def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)
    return boxes, score, label

def get_image_info(path):
    ###coc dataset
    with open(path, 'r')as f:
        eval_file_list = f.read().split('\n')[:-1]
    eval_file_dict = {}
    for i in eval_file_list:
        tmp_list = i.split(' ')
        idx = int(tmp_list[0])
        path = tmp_list[1]
        w = float(tmp_list[2])
        h = float(tmp_list[3])
        bbox_len = len(tmp_list[4:]) // 5
        bbox = []
        for bbox_idx in range(bbox_len):
            label, x1, y1, x2, y2 = tmp_list[4:][bbox_idx * 5:bbox_idx * 5 + 5]
            bbox.append([label, x1, y1, x2, y2])
        eval_file_dict[idx] = {
            'path': path,
            'w': w,
            'h': h,
            'bbox': bbox
        }
    return eval_file_dict

def image_process(eval_path, eval_file_dict):
    ###image process
    args = parse_args()
    imagelist = []
    image_path_list = []
    dw_list = []
    dh_list = []
    resize_ratio_list = []
    if args.max_test > 0:
        test_len = min(args.max_test, len(eval_file_dict.keys()))
    else:
        test_len = len(eval_file_dict.keys())
    for test_idx in trange(test_len):
        image_file = eval_file_dict[test_idx]['path']
        img_ori = cv2.imread(image_file)
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img / 255.
        imagelist.append(img)
        ###obtain images path
        img_path = eval_file_dict[test_idx]['path']
        image_path_list.append(img_path)
        dw_list.append(dw)
        dh_list.append(dh)
        resize_ratio_list.append(resize_ratio)
    return np.array(imagelist),np.array(image_path_list),test_len, dw_list, dh_list, resize_ratio_list

def get_default_dict():
    return {"image_id": -1, "category_id": -1, "bbox": [], "score": 0}

def post_process(boxes_list, scores_list, images_count, image_path_list, eval_file_dict, dw, dh, resize_ratio):
    json_out = []
    args = parse_args()
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)
    batchsize = int(args.batchsize)
    total_step = int(images_count / batchsize)
    cat_id_to_real_id = \
        {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
         18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
         35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
         50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
         64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
         82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
    real_id_to_cat_id = {cat_id_to_real_id[i]: i for i in cat_id_to_real_id}

    for step in range(total_step):
        for j in range(batchsize):
            boxes_ = boxes_list[step][j][np.newaxis,:]
            scores_ = scores_list[step][j][np.newaxis,:]
            boxes_, scores_, labels_ = cpu_nms(boxes_, scores_, args.num_class, args.max_boxes, args.score_thresh,
                                               args.nms_thresh)
            if args.letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw[step * batchsize + j]) / resize_ratio[step * batchsize + j]
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh[step * batchsize + j]) / resize_ratio[step * batchsize + j]
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))
            if args.save_img:
                for i in range(len(boxes_)):
                    x0, y0, x1, y1 = boxes_[i]
                    plot_one_box(img_ori, [x0, y0, x1, y1],
                                 label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                                 color=color_table[labels_[i]])
                cv2.imwrite('tmp/%d_detection_result.jpg' % test_idx, img_ori)
                print('%d done' % test_idx)

            if args.save_json:
                img_path = image_path_list[step * batchsize + j]
                for i in range(len(boxes_)):
                    x0, y0, x1, y1 = boxes_[i]
                    bw = x1 - x0
                    bh = y1 - y0
                    s = scores_[i]
                    c = labels_[i]
                    t_dict = get_default_dict()
                    t_dict['image_id'] = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
                    t_dict['category_id'] = real_id_to_cat_id[int(c) + 1]
                    t_dict['bbox'] = [int(i) for i in [x0, y0, bw, bh]]
                    t_dict['score'] = float(s)
                    json_out.append(t_dict)
    if args.save_json:
        with open(args.save_json_path, 'w')as f:
                json.dump(json_out, f)
        print('output json saved to: ', args.save_json_path)

def get_img_id(file_name):
    import pylab, json
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
      ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def compute_accuary():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    args = parse_args()
    annType = ['segm', 'bbox', 'keypoints']  # set iouType to 'segm', 'bbox' or 'keypoints'
    annType = annType[1]  # specify type here
    cocoGt_file = args.label_file
    cocoGt = COCO(cocoGt_file)
    cocoDt_file = args.save_json_path
    imgIds = get_img_id(cocoDt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)  # image json
    imgIds = sorted(imgIds)  # sort coco image_id
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def main():
    args = parse_args()
    eval_file_dict = get_image_info(args.annotation_txt)
    ###preprocess
    print("########NOW Start Preprocess!!!#########")
    images, image_path, images_count,dw_list, dh_list, resize_ratio_list = image_process(args.annotation_txt, eval_file_dict)

    ###batch preprocess
    print("########NOW Start Batch!!!#########")
    classifier = Classifier()
    batch_images = classifier.batch_process(images)

    ###inference
    print("########NOW Start inference!!!#########")
    boxes_list, scores_list, total_time = classifier.do_infer(batch_images)

    ###post process
    post_process(boxes_list, scores_list, images_count, image_path, eval_file_dict, dw_list, dh_list, resize_ratio_list)

    ###compute accuary MAP
    compute_accuary()

    print('+----------------------------------------+')
    print('images number = ', images_count)
    print('images/sec = ', images_count / total_time)
    print('+----------------------------------------+')

if __name__ == '__main__':
    main()

